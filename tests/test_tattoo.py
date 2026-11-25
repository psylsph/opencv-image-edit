"""Tattoo removal: dark ink inside the selection -> the flesh tone next to it. Nothing else is touched."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.exceptions import ValidationError
from app.pipeline import tattoo as T

SKIN = np.array([120, 150, 205], np.float32)  # BGR
WALL = np.array([200, 120, 60], np.float32)  # a blue-ish background (BGR)
INK = np.array([60, 40, 30], np.float32)  # dark brown-black


def _arm_scene(h=240, w=360, ink_colour=INK, ink_alpha=1.0):
    """A horizontal arm (skin with shading and grain) on a wall, with a dark tattoo in the middle of the arm."""
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:] = WALL
    arm = ((y > 70) & (y < 170)).astype(np.float32)
    shade = 1.0 - 0.15 * ((y - 70) / 100.0).clip(0, 1)
    skin = SKIN[None, None, :] * shade[:, :, None] + rng.normal(0, 3.0, (h, w, 1))
    img = img * (1 - arm[:, :, None]) + skin * arm[:, :, None]
    ink8 = np.zeros((h, w), np.uint8)
    cv2.ellipse(ink8, (180, 120), (45, 24), 15, 0, 360, 255, -1)
    cv2.putText(ink8, "X", (150, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)  # a hole in the ink
    ink = cv2.GaussianBlur(ink8.astype(np.float32) / 255.0, (0, 0), 0.8) * ink_alpha
    img = img * (1 - ink[:, :, None]) + np.asarray(ink_colour, np.float32) * ink[:, :, None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    inked = (ink > 0.5 * ink_alpha).astype(np.uint8)
    # A sloppy brush stroke: covers the tattoo AND runs well off the arm onto the wall (rows 30..210).
    sel = np.zeros((h, w), np.uint8)
    sel[30:210, 110:260] = 1
    return img, sel, inked, arm.astype(np.uint8)


def _lab_dist(a, b):
    la = cv2.cvtColor(np.uint8([[a]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    lb = cv2.cvtColor(np.uint8([[b]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    return float(np.linalg.norm(la - lb))


class TestSensitivity:
    def test_onset_falls_as_sensitivity_rises(self):
        on = [T.onset_for(s, 2.0) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert on == sorted(on, reverse=True) and on[-1] >= T.ONSET_FLOOR

    def test_more_sensitivity_never_finds_less(self):
        # The slider must go the way its label says: higher = catches fainter ink = at least as much ink.
        img, sel, _, _ = _arm_scene(ink_alpha=0.2)
        areas = [float(T.find_ink(img, sel, s)[0].sum()) for s in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        assert areas == sorted(areas), areas

    def test_faint_ink_needs_a_higher_setting(self):
        img, sel, inked, _ = _arm_scene(
            ink_alpha=0.2
        )  # a faded tattoo: only a little off the skin colour
        low = T.find_ink(img, sel, 0.0)[0]
        high = T.find_ink(img, sel, 1.0)[0]
        assert (high * inked).sum() > 3 * max((low * inked).sum(), 1.0)

    def test_more_sensitivity_removes_more_of_a_faint_tattoo(self):
        img, sel, inked, _ = _arm_scene(ink_alpha=0.2)
        m = inked > 0

        def left(s):
            out = T.remove_tattoo(img, sel, grow=s)
            return _lab_dist(out[m].mean(0), SKIN * 0.93)

        assert left(1.0) <= left(0.5) + 1e-6 and left(0.5) < 0.6 * left(0.0)


class TestFindInk:
    def test_finds_the_ink(self):
        img, sel, inked, _ = _arm_scene()
        ink, _ = T.find_ink(img, sel, 0.5)
        assert (ink[inked > 0] > 0.5).mean() > 0.9

    def test_ink_is_only_inside_the_selection(self):
        img, sel, _, _ = _arm_scene()
        tight = np.zeros_like(sel)
        tight[100:140, 140:200] = 1
        ink, _ = T.find_ink(img, tight, 1.0)
        assert ink[tight == 0].sum() == 0

    def test_clean_skin_gives_no_ink(self):
        img, _, _, _ = _arm_scene()
        sel = np.zeros(img.shape[:2], np.uint8)
        sel[85:105, 20:80] = 1
        for s in (0.0, 0.5, 1.0):
            assert T.find_ink(img, sel, s)[0].sum() == 0

    def test_a_grey_wash_that_is_barely_darker_than_the_skin_is_found(self):
        # A soft grey tattoo: no darker than a shadow, only less warm than skin. This is what a faded real one looks like.
        img, sel, inked, _ = _arm_scene(ink_colour=(150, 150, 150), ink_alpha=0.8)
        ink, _ = T.find_ink(img, sel, 0.5)
        assert (ink[inked > 0] > 0.5).mean() > 0.9


class TestRemoveTattoo:
    def test_dark_ink_becomes_flesh_tone(self):
        img, sel, inked, _ = _arm_scene()
        out = T.remove_tattoo(img, sel)
        m = inked > 0
        assert _lab_dist(img[m].mean(0), SKIN) > 40
        assert _lab_dist(out[m].mean(0), SKIN * 0.93) < 12

    def test_the_replacement_is_a_flesh_tone_never_the_backdrop(self):
        img, sel, inked, _ = _arm_scene()
        out = T.remove_tattoo(img, sel)
        fill = out[inked > 0].astype(np.float32)
        assert (
            fill[:, 2].mean() > fill[:, 0].mean() + 30
        )  # warm skin (red >> blue), not the cold blue wall
        assert _lab_dist(fill.mean(0), WALL) > 3 * _lab_dist(fill.mean(0), SKIN * 0.93)

    def test_the_background_inside_the_selection_is_untouched_bit_for_bit(self):
        img, sel, _, arm = _arm_scene()
        out = T.remove_tattoo(img, sel)
        wall = (arm == 0) & (sel > 0)
        assert np.array_equal(out[wall], img[wall])

    def test_pixels_outside_the_selection_are_untouched(self):
        img, sel, _, _ = _arm_scene()
        out = T.remove_tattoo(img, sel)
        assert np.array_equal(out[sel == 0], img[sel == 0])

    def test_fill_has_skin_like_grain(self):
        img, sel, inked, _ = _arm_scene()
        out = T.remove_tattoo(img, sel)

        def grain(a, m):
            l_ = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]
            return float((l_ - cv2.GaussianBlur(l_, (0, 0), 1.5))[m].std())

        inner = cv2.erode(inked, np.ones((3, 3), np.uint8), iterations=4) > 0
        plain = np.zeros_like(inked, bool)
        plain[85:105, 20:80] = True
        assert grain(out, inner) > 0.5 * grain(img, plain)

    def test_strength_zero_is_a_no_op_and_half_is_in_between(self):
        img, sel, inked, _ = _arm_scene()
        assert np.array_equal(T.remove_tattoo(img, sel, strength=0.0), img)
        full, half = (
            T.remove_tattoo(img, sel, strength=1.0),
            T.remove_tattoo(img, sel, strength=0.5),
        )
        m = inked > 0
        d = lambda a: _lab_dist(a[m].mean(0), SKIN * 0.93)  # noqa: E731
        assert d(full) < d(half) < d(img)

    def test_deterministic(self):
        img, sel, _, _ = _arm_scene()
        assert np.array_equal(T.remove_tattoo(img, sel), T.remove_tattoo(img, sel))

    def test_a_strongly_coloured_dark_backdrop_is_not_brown_black_ink(self):
        # A saturated blue wall is darker than skin but is not dark brown / black / grey ink: it must be left alone even
        # when the painted area runs well onto it.
        img, sel, _, arm = _arm_scene()
        out = T.remove_tattoo(img, sel, grow=1.0)
        wall = (arm == 0) & (sel > 0)
        assert np.array_equal(out[wall], img[wall])

    def test_dark_neutral_ink_on_the_arm_edge_next_to_a_dark_backdrop_still_gets_removed(self):
        # Where the tattoo is right at the edge of the skin (next to hair, say) it must still be found: this is the case
        # an overly cautious 'skin on all sides' test used to drop, leaving the tattoo in place.
        img, _, _, arm = _arm_scene()
        img = img.copy()
        img[71:90, 150:230] = (
            45,
            42,
            40,
        )  # black ink hard against the arm's top edge, wall just above
        sel = np.zeros(img.shape[:2], np.uint8)
        sel[60:100, 140:240] = 1
        out, info = T.remove_tattoo_info(img, sel)
        assert info["mode"] == "detected"
        assert _lab_dist(out[75:88, 160:220].reshape(-1, 3).mean(0), SKIN) < 25

    def test_info_reports_how_it_was_found(self):
        img, sel, _, _ = _arm_scene()
        _, info = T.remove_tattoo_info(img, sel)
        assert info["mode"] == "detected" and 0 < info["ink_pct"] < 60

    def test_nothing_dark_in_the_selection_is_reported_and_left_alone(self):
        img, _, _, _ = _arm_scene()
        sel = np.zeros(img.shape[:2], np.uint8)
        sel[85:105, 20:80] = 1
        out, info = T.remove_tattoo_info(img, sel)
        assert info["mode"] == "none" and np.array_equal(out, img) and out is not img

    def test_a_huge_solid_tattoo_is_removed_including_its_middle(self):
        img, _, _, _ = _arm_scene()
        sel = np.zeros(img.shape[:2], np.uint8)
        sel[90:150, 120:250] = 1
        big = img.copy()
        big[95:145, 130:240] = (INK + np.random.default_rng(2).normal(0, 4, (50, 110, 3))).clip(
            0, 255
        )
        out, info = T.remove_tattoo_info(big, sel)
        middle = out[105:135, 150:220].reshape(-1, 3).mean(0)
        assert _lab_dist(middle, SKIN * 0.93) < 15 and info["ink_pct"] > 30

    def test_bad_inputs(self):
        img, sel, _, _ = _arm_scene()
        with pytest.raises(ValidationError):
            T.remove_tattoo(img, sel.astype(np.float32))
        with pytest.raises(ValidationError):
            T.remove_tattoo(img, sel[:10])

    def test_alpha_channel_is_dropped_not_a_crash(self):
        img, sel, _, _ = _arm_scene()
        bgra = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        assert T.remove_tattoo(bgra, sel).shape == img.shape

    def test_a_dark_skinned_arm_with_a_black_tattoo(self):
        dark = np.array([40, 70, 110], np.float32)
        img, sel, inked, arm = _arm_scene()
        recol = img.astype(np.float32)
        recol[(arm > 0) & (inked == 0)] = dark
        recol = np.clip(recol, 0, 255).astype(np.uint8)
        recol[inked > 0] = (10, 10, 12)
        out = T.remove_tattoo(recol, sel, grow=0.7)
        assert _lab_dist(out[inked > 0].mean(0), dark) < 20


class TestPersonMask:
    """With the person from background matting, the backdrop is off limits however skin-like it is."""

    @staticmethod
    def _scene():
        # An arm in front of a *skin-coloured* wall with a dark band (a shadow) across it, plus a dark tattoo on the arm.
        rng = np.random.default_rng(5)
        h, w = 240, 360
        img = np.empty((h, w, 3), np.float32)
        img[:] = (150, 170, 220)  # a lighter, pinker "skin-like" wall (BGR)
        arm = np.zeros((h, w), np.uint8)
        arm[80:170, :] = 1
        skin = np.array([110, 140, 195], np.float32)[None, None, :] + rng.normal(0, 2.5, (h, w, 1))
        img = np.where(arm[..., None] > 0, skin, img)
        img[20:50, :] = (70, 80, 90)  # a dark neutral shadow band on the wall, above the arm
        ink = np.zeros((h, w), np.uint8)
        cv2.ellipse(ink, (180, 125), (40, 14), 0, 0, 360, 255, -1)
        img[ink > 0] = (45, 42, 40)
        sel = np.zeros((h, w), np.uint8)
        sel[10:180, 100:270] = 1  # covers the shadow band, the wall and the arm with the tattoo
        return np.clip(img, 0, 255).astype(np.uint8), sel, ink > 0, arm

    def test_without_the_person_mask_the_tattoo_is_still_removed(self):
        # The tool works without the matting model: it just has no backdrop protection beyond colour and brightness.
        img, sel, ink, arm = self._scene()
        out = T.remove_tattoo(img, sel)
        skin = np.array([110, 140, 195], np.float32)
        assert _lab_dist(out[ink].mean(0), skin) < 15

    def test_with_the_person_mask_the_backdrop_is_never_touched(self):
        img, sel, ink, arm = self._scene()
        out = T.remove_tattoo(img, sel, foreground=arm)
        off_person = (arm == 0) & (sel > 0)
        assert np.array_equal(
            out[off_person], img[off_person]
        )  # not a single backdrop pixel changed

    def test_with_the_person_mask_the_tattoo_is_still_removed_with_skin_from_the_arm(self):
        img, sel, ink, arm = self._scene()
        out = T.remove_tattoo(img, sel, foreground=arm)
        skin = np.array([110, 140, 195], np.float32)
        assert (
            _lab_dist(out[ink].mean(0), skin) < 12
        )  # the arm's skin colour, not the pinker wall's

    def test_the_skin_colour_is_never_sampled_from_the_backdrop(self):
        # The wall here is the most common colour in the selection and also skin-toned. Without the mask it can win;
        # with it, the fill must match the arm, not the wall.
        img, sel, ink, arm = self._scene()
        with_mask = T.remove_tattoo(img, sel, foreground=arm)
        wall = np.array([150, 170, 220], np.float32)
        arm_skin = np.array([110, 140, 195], np.float32)
        d_arm = _lab_dist(with_mask[ink].mean(0), arm_skin)
        d_wall = _lab_dist(with_mask[ink].mean(0), wall)
        assert d_arm < d_wall

    def test_an_empty_person_mask_inside_the_selection_changes_nothing(self):
        img, sel, _, _ = self._scene()
        out, info = T.remove_tattoo_info(img, sel, foreground=np.zeros(img.shape[:2], np.uint8))
        assert info["mode"] == "none" and np.array_equal(out, img)


class TestShadedSkinIsNotInk:
    """Only the laser-like pixels are replaced: skin shadows, creases and freckles keep their pixels."""

    @staticmethod
    def _scene():
        img, sel, inked, arm = _arm_scene()
        img = img.copy()
        rng = np.random.default_rng(9)
        # Skin features inside the painted area: a soft shadow band, a crease, and some freckles (all skin-hued, darker).
        shade = np.ones(img.shape[:2], np.float32)
        shade[150:160, 110:260] = 0.72  # a shadow along the arm's lower edge
        shade[80:83, 120:170] = 0.66  # a thin crease
        for _ in range(25):
            y, x = rng.integers(75, 165), rng.integers(115, 255)
            if not inked[max(0, y - 6) : y + 6, max(0, x - 6) : x + 6].any():
                shade[y : y + 2, x : x + 2] = 0.7
        arm_px = arm > 0
        out = img.astype(np.float32)
        out[arm_px] *= shade[arm_px][:, None]
        return np.clip(out, 0, 255).astype(np.uint8), sel, inked, arm

    def test_shadows_creases_and_freckles_are_left_alone(self):
        img, sel, inked, arm = self._scene()
        out = T.remove_tattoo(img, sel, grow=0.5)
        features = np.zeros(img.shape[:2], bool)
        features[150:160, 110:260] = True
        features[80:83, 120:170] = True
        features &= arm > 0
        changed = (out[features] != img[features]).any(axis=1)
        assert changed.mean() < 0.05  # at most the rows touching the tattoo's own soft edge

    def test_pixels_that_are_not_ink_are_not_changed_at_all(self):
        img, sel, inked, arm = self._scene()
        out = T.remove_tattoo(img, sel, grow=0.5)
        near_ink = cv2.dilate(inked.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=4) > 0
        away = (sel > 0) & ~near_ink
        changed = (np.abs(out.astype(int) - img.astype(int)).sum(2) > 6) & away
        assert (
            changed.sum() / max(away.sum(), 1) < 0.01
        )  # under 1% of the painted area away from the tattoo

    def test_the_tattoo_itself_is_still_removed(self):
        img, sel, inked, _ = self._scene()
        out = T.remove_tattoo(img, sel, grow=0.5)
        skin = np.array([110, 140, 195], np.float32)
        assert _lab_dist(out[inked > 0].mean(0), skin * 0.93) < 15


class TestHelpers:
    def test_skin_model_follows_brightness(self):
        # Skin gets a little warmer as it gets darker; the model must follow that and not average it away.
        rng = np.random.default_rng(1)
        lab = np.zeros((80, 80, 3), np.float32)
        lab[..., 0] = rng.uniform(90, 190, (80, 80))
        lab[..., 1] = 150 - 0.1 * (lab[..., 0] - 90) + rng.normal(0, 0.5, (80, 80))
        lab[..., 2] = 150 + rng.normal(0, 0.5, (80, 80))
        model = T.SkinModel(lab, np.ones((80, 80), bool))
        a_dark, _ = model.reference(np.array([100.0]))
        a_lit, _ = model.reference(np.array([180.0]))
        assert a_dark[0] > a_lit[0] + 5

    def test_envelope_removes_dark_features_but_keeps_the_shading(self):
        y, x = np.mgrid[0:200, 0:300].astype(np.float32)
        base = 200.0 - 0.2 * x  # a smooth gradient: the limb's shading
        lum = base.copy()
        lum[80:120, 100:160] -= 60  # a dark blob narrower than the structuring element
        env = T._envelope(lum, np.ones(lum.shape, bool), 81)
        assert abs(float(env[100, 130]) - float(base[100, 130])) < 6
        assert float(env[100, 10]) > float(env[100, 290]) + 30


class TestEndpoint:
    @staticmethod
    def _post(client, img, mask, **data):
        files = {
            "file": ("a.png", cv2.imencode(".png", img)[1].tobytes(), "image/png"),
            "mask": ("m.png", cv2.imencode(".png", mask * 255)[1].tobytes(), "image/png"),
        }
        return client.post("/api/v1/tattoo", files=files, data=data)

    def test_removes_and_returns_a_png(self, test_client: TestClient):
        img, sel, inked, arm = _arm_scene()
        r = self._post(test_client, img, sel)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mode"] == "detected" and body["ink_pct"] > 0
        assert body["output_size"] == {"width": 360, "height": 240}
        out = cv2.imdecode(
            np.frombuffer(base64.b64decode(body["final"].split(",", 1)[1]), np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert _lab_dist(out[inked > 0].mean(0), SKIN * 0.93) < 12
        wall = (arm == 0) & (sel > 0)
        assert np.array_equal(out[wall], img[wall])

    def test_an_old_client_still_sending_algorithm_works(self, test_client: TestClient):
        img, sel, _, _ = _arm_scene()
        assert self._post(test_client, img, sel, algorithm="lama").status_code == 200

    def test_empty_mask_is_422(self, test_client: TestClient):
        img, sel, _, _ = _arm_scene()
        assert self._post(test_client, img, np.zeros_like(sel)).status_code == 422

    def test_mask_size_mismatch_is_400(self, test_client: TestClient):
        img, sel, _, _ = _arm_scene()
        assert self._post(test_client, img, sel[:100]).status_code == 400

    def test_out_of_range_options_are_422(self, test_client: TestClient):
        img, sel, _, _ = _arm_scene()
        assert self._post(test_client, img, sel, strength=1.5).status_code == 422
        assert self._post(test_client, img, sel, grow=-0.1).status_code == 422

    def test_garbage_image_is_400(self, test_client: TestClient):
        _, sel, _, _ = _arm_scene()
        r = test_client.post(
            "/api/v1/tattoo",
            files={
                "file": ("a.png", b"junk", "image/png"),
                "mask": ("m.png", cv2.imencode(".png", sel * 255)[1].tobytes(), "image/png"),
            },
        )
        assert r.status_code == 400


class _StubLaMa:
    """Records what the model was given; 'inpaints' the hole with the mean of the unmasked pixels."""

    def __init__(self):
        self.seen = None
        self.iterations = None

    def infer(self, img, mask, iterations=1, **_):
        self.seen, self.iterations = img.copy(), iterations
        out = img.copy()
        out[mask > 0] = img[mask == 0].mean(axis=0).astype(np.uint8)
        return out


class TestLamaTattoo:
    def _run(self, monkeypatch, **kw):
        from app.models.lama import LaMa

        stub = _StubLaMa()
        monkeypatch.setattr(LaMa, "get", classmethod(lambda cls, *a, **k: stub))
        img, sel, _, arm = _arm_scene()
        out, info = T.remove_tattoo_lama(img, sel, foreground=arm, **kw)
        return img, sel, arm, out, info, stub

    def test_runs_twenty_passes_and_hides_the_backdrop_from_the_model(self, monkeypatch):
        img, _, arm, _, info, stub = self._run(monkeypatch)
        assert stub.iterations == 20 and info["mode"] == "lama" and info["passes"] == 20
        # Where the model looked, the wall is gone: every backdrop pixel it saw is skin-coloured, not wall-blue.
        crop_arm = arm[: stub.seen.shape[0], : stub.seen.shape[1]]
        wall_seen = stub.seen[crop_arm == 0].astype(np.float32)
        assert np.abs(wall_seen - WALL).mean() > 40

    def test_backdrop_is_restored_and_only_selected_skin_changes(self, monkeypatch):
        img, sel, arm, out, _, _ = self._run(monkeypatch)
        assert np.array_equal(
            out[arm == 0], img[arm == 0]
        )  # wall untouched, even under the sloppy selection
        assert np.array_equal(out[sel == 0][:100], img[sel == 0][:100])
        assert (out[(sel > 0) & (arm > 0)] != img[(sel > 0) & (arm > 0)]).any()

    def test_strength_zero_changes_nothing(self, monkeypatch):
        img, sel, _, _, _, _ = self._run(monkeypatch)
        from app.pipeline.tattoo import remove_tattoo_lama

        out, info = remove_tattoo_lama(img, sel, strength=0.0)
        assert np.array_equal(out, img) and info["mode"] == "none"

    def test_temporary_background_keeps_the_person_and_replaces_the_rest(self):
        img, _, _, arm = _arm_scene()
        tmp = T._temporary_background(img, arm)
        assert np.array_equal(tmp[arm > 0], img[arm > 0])
        assert np.abs(tmp[arm == 0].astype(np.float32) - WALL).mean() > 40
