"""Tattoo removal: skin sampling, catching faint ink, texture, blending, and the endpoint (TELEA fill, no models)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.exceptions import ValidationError
from app.pipeline import tattoo as T

SKIN = np.array([120, 150, 205], np.float32)  # BGR


def _scene(h=240, w=320):
    """Skin with a soft shading gradient and fine grain, plus a dark tattoo with a faint halo and a thin outline."""
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 1.0 - 0.12 * (x / w)
    img = SKIN[None, None, :] * shade[:, :, None] + rng.normal(0, 3.0, (h, w, 1))
    core = np.zeros((h, w), np.float32)
    cv2.ellipse(core, (160, 120), (40, 28), 20, 0, 360, 1.0, -1)
    halo = (
        cv2.dilate(core, np.ones((3, 3), np.uint8), iterations=6) - core
    )  # faint ring around the ink
    img = img * (1 - core[:, :, None]) + np.array([60, 40, 30], np.float32) * core[:, :, None]
    img = (
        img * (1 - 0.5 * halo[:, :, None])
        + np.array([90, 100, 140], np.float32) * 0.5 * halo[:, :, None]
    )
    img = np.clip(img, 0, 255).astype(np.uint8)
    selection = cv2.erode(
        core.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=5
    )  # misses the ink edge
    return img, selection, core.astype(np.uint8), halo.astype(np.uint8)


def _lab_dist(a, b):
    la = cv2.cvtColor(np.uint8([[a]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    lb = cv2.cvtColor(np.uint8([[b]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    return float(np.linalg.norm(la - lb))


class TestSkinSample:
    def test_median_matches_the_skin_not_the_ink(self):
        img, sel, _, _ = _scene()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        med, spread = T.skin_sample(lab, sel, 4)
        skin_lab = cv2.cvtColor(np.uint8([[SKIN]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        assert np.linalg.norm(med - skin_lab) < 12 and spread < 8

    def test_whole_picture_selected_is_an_error(self):
        lab = np.zeros((20, 20, 3), np.float32)
        with pytest.raises(ValidationError):
            T.skin_sample(lab, np.ones((20, 20), np.uint8), 2)


class TestGrow:
    def test_grow_catches_ink_the_selection_missed(self):
        img, sel, core, _ = _scene()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        med, spread = T.skin_sample(lab, sel, 4)
        none = T.grow_selection(lab, sel, med, spread, 0.0, 4)
        lots = T.grow_selection(lab, sel, med, spread, 1.0, 4)
        assert (lots & core).sum() > (none & core).sum()
        assert lots.sum() >= none.sum()

    def test_growth_is_bounded(self):
        img, sel, _, _ = _scene()
        # An image that is mostly "not skin" must not make the mask swallow it.
        img[:, :] = (20, 20, 20)
        img[100:140, 140:180] = SKIN.astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        med = np.array([150, 140, 150], np.float32)
        grown = T.grow_selection(lab, sel, med, 2.0, 1.0, 4)
        assert grown.sum() <= T.MAX_GROWTH * sel.sum()


class TestRemoveTattoo:
    def test_tattoo_disappears(self):
        img, sel, core, _ = _scene()
        out = T.remove_tattoo(img, sel, grow=1.0, algorithm="telea")
        inside = core > 0
        assert _lab_dist(img[inside].mean(0), SKIN) > 40  # clearly inked before
        assert _lab_dist(out[inside].mean(0), SKIN * 0.94) < 14  # looks like skin after

    def test_faint_halo_is_caught_with_grow(self):
        img, sel, _, halo = _scene()
        h = halo > 0
        skin = img[:30, :30].reshape(-1, 3).mean(0)
        off = T.remove_tattoo(img, sel, grow=0.0, algorithm="telea")
        on = T.remove_tattoo(img, sel, grow=1.0, algorithm="telea")
        assert _lab_dist(on[h].mean(0), skin) < _lab_dist(off[h].mean(0), skin)

    def test_pixels_far_away_are_untouched(self):
        img, sel, _, _ = _scene()
        out = T.remove_tattoo(img, sel, algorithm="telea")
        assert np.array_equal(out[:20], img[:20]) and np.array_equal(out[:, :40], img[:, :40])

    def test_strength_zero_is_a_no_op_and_half_is_in_between(self):
        img, sel, core, _ = _scene()
        assert np.array_equal(T.remove_tattoo(img, sel, strength=0.0, algorithm="telea"), img)
        full = T.remove_tattoo(img, sel, strength=1.0, algorithm="telea")
        half = T.remove_tattoo(img, sel, strength=0.5, algorithm="telea")
        c = core > 0
        d = lambda a: _lab_dist(a[c].mean(0), SKIN * 0.94)  # noqa: E731
        assert d(full) < d(half) < d(img)

    def test_fill_has_skin_like_grain_not_a_smooth_patch(self):
        img, sel, core, _ = _scene()
        out = T.remove_tattoo(img, sel, algorithm="telea")

        def grain(a, m):
            l_ = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]
            return float((l_ - cv2.GaussianBlur(l_, (0, 0), 1.5))[m].std())

        inner = cv2.erode(core, np.ones((3, 3), np.uint8), iterations=8) > 0
        outer = np.zeros_like(core, bool)
        outer[:60, :60] = True
        assert grain(out, inner) > 0.5 * grain(img, outer)  # a smooth fill would be near 0

    def test_deterministic(self):
        img, sel, _, _ = _scene()
        a = T.remove_tattoo(img, sel, algorithm="telea")
        b = T.remove_tattoo(img, sel, algorithm="telea")
        assert np.array_equal(a, b)

    def test_empty_selection_returns_a_copy(self):
        img, _, _, _ = _scene()
        out = T.remove_tattoo(img, np.zeros(img.shape[:2], np.uint8), algorithm="telea")
        assert np.array_equal(out, img) and out is not img

    def test_bad_inputs(self):
        img, sel, _, _ = _scene()
        with pytest.raises(ValidationError):
            T.remove_tattoo(img, sel.astype(np.float32))
        with pytest.raises(ValidationError):
            T.remove_tattoo(img, sel[:10])

    def test_alpha_channel_is_dropped_not_a_crash(self):
        img, sel, _, _ = _scene()
        bgra = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        assert T.remove_tattoo(bgra, sel, algorithm="telea").shape == img.shape


class TestEndpoint:
    @staticmethod
    def _post(client, img, mask, **data):
        files = {
            "file": ("a.png", cv2.imencode(".png", img)[1].tobytes(), "image/png"),
            "mask": ("m.png", cv2.imencode(".png", mask * 255)[1].tobytes(), "image/png"),
        }
        return client.post("/api/v1/tattoo", files=files, data={"algorithm": "telea", **data})

    def test_removes_and_returns_a_png(self, test_client: TestClient):
        import base64

        img, sel, core, _ = _scene()
        r = self._post(test_client, img, sel, grow="1.0")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["output_size"] == {"width": 320, "height": 240}
        out = cv2.imdecode(
            np.frombuffer(base64.b64decode(body["final"].split(",", 1)[1]), np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert _lab_dist(out[core > 0].mean(0), SKIN * 0.94) < 14

    def test_empty_mask_is_422(self, test_client: TestClient):
        img, sel, _, _ = _scene()
        assert self._post(test_client, img, np.zeros_like(sel)).status_code == 422

    def test_mask_size_mismatch_is_400(self, test_client: TestClient):
        img, sel, _, _ = _scene()
        assert self._post(test_client, img, sel[:100]).status_code == 400

    def test_out_of_range_options_are_422(self, test_client: TestClient):
        img, sel, _, _ = _scene()
        assert self._post(test_client, img, sel, strength=1.5).status_code == 422
        assert self._post(test_client, img, sel, grow=-0.1).status_code == 422

    def test_garbage_image_is_400(self, test_client: TestClient):
        _, sel, _, _ = _scene()
        r = test_client.post(
            "/api/v1/tattoo",
            files={
                "file": ("a.png", b"junk", "image/png"),
                "mask": ("m.png", cv2.imencode(".png", sel * 255)[1].tobytes(), "image/png"),
            },
        )
        assert r.status_code == 400
