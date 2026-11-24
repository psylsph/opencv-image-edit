"""Face replacement: geometry (no models), engine error handling, and the API with a stubbed engine."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.exceptions import ValidationError
from app.models import face as face_models
from app.models.face import Face, FaceEngine, face_models_status
from app.pipeline import face as geo


def _kps(cx=100.0, cy=100.0, scale=1.0, angle=0.0):
    """ArcFace landmarks placed at (cx, cy) with a given scale/rotation, as image-space keypoints."""
    t = geo.ARCFACE_112 - geo.ARCFACE_112.mean(0)
    a = np.deg2rad(angle)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (t @ r.T * scale + (cx, cy)).astype(np.float32)


class TestSimilarity:
    def test_recovers_a_known_transform(self):
        src = geo.ARCFACE_112
        a = np.deg2rad(20)
        true = np.array(
            [[1.7 * np.cos(a), -1.7 * np.sin(a), 30], [1.7 * np.sin(a), 1.7 * np.cos(a), -12]],
            np.float32,
        )
        dst = src @ true[:, :2].T + true[:, 2]
        m = geo.similarity_transform(src, dst)
        assert np.allclose(m, true, atol=1e-3)

    def test_never_shears(self):
        m = geo.similarity_transform(_kps(angle=15, scale=2.0), geo.arcface_template(128))
        # a similarity has the form [[a, -b], [b, a]]
        assert m[0, 0] == pytest.approx(m[1, 1], abs=1e-4)
        assert m[0, 1] == pytest.approx(-m[1, 0], abs=1e-4)

    def test_degenerate_landmarks_rejected(self):
        with pytest.raises(ValueError):
            geo.similarity_transform(np.zeros((5, 2)), geo.ARCFACE_112)

    def test_align_crop_puts_landmarks_on_the_template(self):
        img = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
        kps = _kps(150, 140, scale=2.0, angle=10)
        crop, m = geo.align_crop(img, kps, geo.arcface_template(128), 128)
        assert crop.shape == (128, 128, 3)
        mapped = kps @ m[:, :2].T + m[:, 2]
        assert np.abs(mapped - geo.arcface_template(128)).max() < 0.5

    def test_templates_scale(self):
        assert np.allclose(geo.arcface_template(224), geo.ARCFACE_112 * 2)
        assert geo.ffhq_template(512).max() < 512


class TestMasks:
    def test_range_centre_and_corners(self):
        m = geo.face_mask(128, 0.5)
        assert m.shape == (128, 128) and m.dtype == np.float32
        assert 0.0 <= m.min() and m.max() <= 1.0
        assert m[64, 64] == pytest.approx(1.0, abs=1e-3)
        assert m[0, 0] == 0.0 and m[-1, -1] == 0.0 and m[0, -1] == 0.0

    def test_edge_never_touches_the_crop_border(self):
        # A visible line at the chin appeared when the fade reached the crop's last rows.
        for feather in (0.0, 0.5, 1.0):
            m = geo.face_mask(128, feather)
            assert m[-1].max() < 0.05 and m[0].max() < 0.05
            assert m[:, 0].max() < 0.05 and m[:, -1].max() < 0.05
            r = geo.ffhq_face_mask(512, feather)
            assert r[-1].max() < 0.05 and r[:, 0].max() < 0.05

    def test_feather_widens_the_transition(self):
        hard, soft = geo.face_mask(128, 0.0), geo.face_mask(128, 1.0)
        assert ((hard > 0.05) & (hard < 0.95)).sum() < ((soft > 0.05) & (soft < 0.95)).sum()


class TestColourMatch:
    def _pair(self):
        rng = np.random.default_rng(1)
        ref = np.clip(rng.normal((90, 120, 170), 10, (64, 64, 3)), 0, 255).astype(
            np.uint8
        )  # warm skin
        src = np.clip(rng.normal((170, 140, 110), 10, (64, 64, 3)), 0, 255).astype(np.uint8)  # cool
        return src, ref, np.ones((64, 64), np.float32)

    def test_amount_zero_is_a_no_op(self):
        src, ref, mask = self._pair()
        assert geo.match_color(src, ref, mask, 0.0) is src

    def test_full_match_moves_the_mean_to_the_reference(self):
        src, ref, mask = self._pair()
        out = geo.match_color(src, ref, mask, 1.0)
        assert np.abs(out.reshape(-1, 3).mean(0) - ref.reshape(-1, 3).mean(0)).max() < 4
        assert np.abs(out.reshape(-1, 3).mean(0) - src.reshape(-1, 3).mean(0)).max() > 20

    def test_half_match_is_in_between(self):
        src, ref, mask = self._pair()
        full, half = geo.match_color(src, ref, mask, 1.0), geo.match_color(src, ref, mask, 0.5)
        s, f, h = (x.reshape(-1, 3).mean(0) for x in (src, full, half))
        assert np.all(np.minimum(s, f) - 3 <= h) and np.all(h <= np.maximum(s, f) + 3)

    def test_tiny_mask_is_ignored(self):
        src, ref, _ = self._pair()
        assert geo.match_color(src, ref, np.zeros((64, 64), np.float32), 1.0) is src


class TestPasteBack:
    def test_pasting_the_same_crop_changes_almost_nothing(self):
        rng = np.random.default_rng(2)
        img = cv2.GaussianBlur(rng.integers(0, 255, (240, 260, 3), dtype=np.uint8), (0, 0), 3)
        kps = _kps(130, 120, scale=1.5, angle=8)
        crop, m = geo.align_crop(img, kps, geo.arcface_template(128), 128)
        out = geo.paste_back(img, crop, m, geo.face_mask(128, 0.5))
        assert out.shape == img.shape and out is not img
        assert np.abs(out.astype(int) - img.astype(int)).mean() < 2.0

    def test_only_the_face_area_changes(self):
        img = np.full((300, 300, 3), 100, np.uint8)
        kps = _kps(150, 150, scale=1.0)
        _, m = geo.align_crop(img, kps, geo.arcface_template(128), 128)
        patch = np.full((128, 128, 3), 250, np.uint8)
        out = geo.paste_back(img, patch, m, geo.face_mask(128, 0.3))
        changed = np.abs(out.astype(int) - img.astype(int)).sum(2) > 0
        ys, xs = np.nonzero(changed)
        assert changed[150, 150] and out[150, 150, 0] > 240
        assert (
            xs.min() > 150 - 80
            and xs.max() < 150 + 80
            and ys.min() > 150 - 90
            and ys.max() < 150 + 90
        )
        assert not changed[0, 0] and not changed[-1, -1]

    def test_opacity_scales_the_effect(self):
        img = np.full((200, 200, 3), 100, np.uint8)
        _, m = geo.align_crop(img, _kps(100, 100), geo.arcface_template(128), 128)
        patch = np.full((128, 128, 3), 200, np.uint8)
        out = geo.paste_back(img, patch, m, geo.face_mask(128, 0.0), opacity=0.5)
        assert 140 <= out[100, 100, 0] <= 160
        assert np.array_equal(
            geo.paste_back(img, patch, m, geo.face_mask(128, 0.0), opacity=0.0), img
        )

    def test_face_at_the_image_edge_is_clipped_not_an_error(self):
        img = np.full((120, 120, 3), 100, np.uint8)
        _, m = geo.align_crop(img, _kps(5, 5, scale=1.5), geo.arcface_template(128), 128)
        out = geo.paste_back(img, np.full((128, 128, 3), 250, np.uint8), m, geo.face_mask(128, 0.3))
        assert out.shape == img.shape

    def test_face_entirely_outside_is_a_no_op(self):
        img = np.full((100, 100, 3), 100, np.uint8)
        _, m = geo.align_crop(img, _kps(900, 900), geo.arcface_template(128), 128)
        out = geo.paste_back(img, np.full((128, 128, 3), 250, np.uint8), m, geo.face_mask(128))
        assert np.array_equal(out, img)


class TestEngineErrors:
    def _engine(self, monkeypatch, target_faces, donor_faces):
        engine = FaceEngine("/nonexistent")
        faces = {"t": target_faces, "d": donor_faces}
        monkeypatch.setattr(
            engine, "detect", lambda img: faces["t"] if img.shape[0] == 50 else faces["d"]
        )
        return engine

    def _face(self, x=10.0):
        return Face((x, 10.0, 30.0, 30.0), _kps(x + 15, 25), 0.9)

    def test_no_face_in_photo(self, monkeypatch):
        e = self._engine(monkeypatch, [], [self._face()])
        with pytest.raises(ValidationError, match="no face found in the photo"):
            e.swap(np.zeros((50, 50, 3), np.uint8), np.zeros((60, 60, 3), np.uint8))

    def test_no_face_in_donor(self, monkeypatch):
        e = self._engine(monkeypatch, [self._face()], [])
        with pytest.raises(ValidationError, match="no face found in the donor"):
            e.swap(np.zeros((50, 50, 3), np.uint8), np.zeros((60, 60, 3), np.uint8))

    def test_index_out_of_range(self, monkeypatch):
        e = self._engine(monkeypatch, [self._face()], [self._face()])
        with pytest.raises(ValidationError, match="face 3 does not exist"):
            e.swap(np.zeros((50, 50, 3), np.uint8), np.zeros((60, 60, 3), np.uint8), target_index=2)
        with pytest.raises(ValidationError, match="donor photo has 1 face"):
            e.swap(np.zeros((50, 50, 3), np.uint8), np.zeros((60, 60, 3), np.uint8), donor_index=1)

    def test_missing_model_files_raise_a_clear_503_error(self, monkeypatch, tmp_path):
        from app.exceptions import ModelNotFoundError

        e = FaceEngine(tmp_path)
        with pytest.raises(ModelNotFoundError, match="inswapper|arcface|yunet"):
            e.detect(np.zeros((50, 50, 3), np.uint8))

    def test_status_lists_missing_files(self, tmp_path):
        s = face_models_status(tmp_path)
        assert not s["ready"] and not s["restorer"] and len(s["missing"]) == 3
        for info in face_models.FACE_MODELS.values():
            (tmp_path / info.filename).write_bytes(b"x")
        assert face_models_status(tmp_path) == {"ready": True, "restorer": True, "missing": []}


class _StubEngine:
    """Detects one fixed face per image and 'swaps' by painting the face area white, so the pipeline is visible."""

    def __init__(self):
        self.calls = []

    def detect(self, img):
        h, w = img.shape[:2]
        return [Face((w * 0.3, h * 0.3, w * 0.4, h * 0.4), _kps(w / 2, h / 2, scale=w / 200), 0.95)]

    def swap(self, target, donor, **kw):
        self.calls.append(kw)
        out = target.copy()
        h, w = out.shape[:2]
        out[int(h * 0.4) : int(h * 0.6), int(w * 0.4) : int(w * 0.6)] = 255
        return out


class TestApi:
    @pytest.fixture(autouse=True)
    def stub(self, monkeypatch):
        self.engine = _StubEngine()
        monkeypatch.setattr("app.models.face.get_face_engine", lambda *a, **k: self.engine)
        monkeypatch.setattr("app.api.faces.get_face_engine", lambda *a, **k: self.engine)

    @staticmethod
    def _png(arr):
        return cv2.imencode(".png", arr)[1].tobytes()

    def _post(self, client, face=None, donor=True, size=(80, 100)):
        img = np.full((*size, 3), (10, 20, 30), np.uint8)
        files = {"file": ("a.png", self._png(img), "image/png")}
        if donor:
            files["face_donor"] = (
                "d.png",
                self._png(np.full((60, 60, 3), 120, np.uint8)),
                "image/png",
            )
        settings = {"face": {"enabled": True, **(face or {})}}
        return client.post("/api/v1/process", files=files, data={"settings": json.dumps(settings)})

    @staticmethod
    def _final(resp):
        import base64

        raw = base64.b64decode(resp.json()["final"].split(",", 1)[1])
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    def test_faces_endpoint(self, test_client: TestClient):
        r = test_client.post(
            "/api/v1/faces",
            files={"file": ("a.png", self._png(np.zeros((80, 100, 3), np.uint8)), "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert (body["width"], body["height"]) == (100, 80)
        assert len(body["faces"]) == 1 and len(body["faces"][0]["landmarks"]) == 5
        assert body["faces"][0]["box"] == [30.0, 24.0, 40.0, 32.0]

    def test_faces_endpoint_rejects_garbage(self, test_client: TestClient):
        r = test_client.post(
            "/api/v1/faces", files={"file": ("a.png", b"not an image", "image/png")}
        )
        assert r.status_code == 400

    def test_faces_status_endpoint(self, test_client: TestClient):
        r = test_client.get("/api/v1/faces/status")
        assert r.status_code == 200 and set(r.json()) == {"ready", "restorer", "missing"}

    def test_process_swaps_the_face(self, test_client: TestClient):
        r = self._post(
            test_client, {"target": 0, "donor": 0, "blend": 0.2, "color_match": 0.1, "restore": 0.4}
        )
        assert r.status_code == 200, r.text
        out = self._final(r)
        assert (
            out[40, 50].tolist() == [255, 255, 255]
            and out[2, 2].tolist() == [30, 20, 10]
            or out[2, 2].tolist() == [10, 20, 30]
        )
        assert self.engine.calls == [
            {"target_index": 0, "donor_index": 0, "blend": 0.2, "color_match": 0.1, "restore": 0.4}
        ]

    def test_disabled_face_does_not_touch_the_engine(self, test_client: TestClient):
        img = np.full((80, 100, 3), 5, np.uint8)
        r = test_client.post(
            "/api/v1/process",
            files={"file": ("a.png", self._png(img), "image/png")},
            data={"settings": json.dumps({"face": {"enabled": False}})},
        )
        assert r.status_code == 200 and self.engine.calls == []

    def test_missing_donor_is_422(self, test_client: TestClient):
        assert self._post(test_client, donor=False).status_code == 422

    def test_bad_donor_is_400(self, test_client: TestClient):
        img = np.full((80, 100, 3), 5, np.uint8)
        r = test_client.post(
            "/api/v1/process",
            files={
                "file": ("a.png", self._png(img), "image/png"),
                "face_donor": ("d.png", b"junk", "image/png"),
            },
            data={"settings": json.dumps({"face": {"enabled": True}})},
        )
        assert r.status_code == 400

    def test_out_of_range_settings_are_422(self, test_client: TestClient):
        assert self._post(test_client, {"blend": 2}).status_code == 422
        assert self._post(test_client, {"target": -1}).status_code == 422

    def test_no_face_found_is_a_422_with_the_reason(self, test_client: TestClient, monkeypatch):
        def boom(*a, **k):
            raise ValidationError("no face found in the donor photo")

        monkeypatch.setattr(self.engine, "swap", boom)
        r = self._post(test_client)
        assert r.status_code == 422 and "no face found in the donor photo" in r.text

    def test_face_runs_before_background(self, test_client: TestClient, monkeypatch):
        order = []
        monkeypatch.setattr(self.engine, "swap", lambda t, d, **k: (order.append("face"), t)[1])

        class M:
            def predict(self, img):
                order.append("matting")
                return np.ones(img.shape[:2], np.float32)

        monkeypatch.setattr("app.pipeline.get_matting_model", lambda name: M())
        img = np.full((80, 100, 3), 5, np.uint8)
        files = {
            "file": ("a.png", self._png(img), "image/png"),
            "face_donor": ("d.png", self._png(img), "image/png"),
        }
        settings = {"face": {"enabled": True}, "background": {"enabled": True, "mode": "remove"}}
        r = test_client.post(
            "/api/v1/process", files=files, data={"settings": json.dumps(settings)}
        )
        assert r.status_code == 200 and order == ["face", "matting"]
