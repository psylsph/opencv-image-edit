"""Tattoo removal: the selection is only the search area; ink on skin is replaced with surrounding skin."""

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
INK = np.array([60, 40, 30], np.float32)


def _arm_scene(h=240, w=360):
    """A horizontal arm (skin with shading and grain) on a wall, with a tattoo in the middle of the arm."""
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:] = WALL
    arm = ((y > 70) & (y < 170)).astype(np.float32)  # the arm: rows 70..170
    shade = 1.0 - 0.15 * ((y - 70) / 100.0).clip(0, 1)
    skin = SKIN[None, None, :] * shade[:, :, None] + rng.normal(0, 3.0, (h, w, 1))
    img = img * (1 - arm[:, :, None]) + skin * arm[:, :, None]
    ink8 = np.zeros((h, w), np.uint8)
    cv2.ellipse(ink8, (180, 120), (45, 24), 15, 0, 360, 255, -1)
    cv2.putText(ink8, "X", (150, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)  # a hole in the ink
    ink = cv2.GaussianBlur(ink8.astype(np.float32) / 255.0, (0, 0), 0.8)
    img = img * (1 - ink[:, :, None]) + INK * ink[:, :, None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    inked = (ink > 0.5).astype(np.uint8)
    # A sloppy brush stroke: covers the tattoo AND runs well off the arm onto the wall (rows 30..210).
    sel = np.zeros((h, w), np.uint8)
    sel[30:210, 110:260] = 1
    return img, sel, inked, arm.astype(np.uint8)


def _lab_dist(a, b):
    la = cv2.cvtColor(np.uint8([[a]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    lb = cv2.cvtColor(np.uint8([[b]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    return float(np.linalg.norm(la - lb))


class TestFindInk:
    def test_ink_is_found_and_background_is_not(self):
        img, sel, inked, arm = _arm_scene()
        ink, skin, _ = T.find_ink(img, sel, 0.3)
        assert (ink & inked).sum() > 0.9 * inked.sum()  # nearly all the tattoo
        off_arm = (arm == 0) & (sel > 0)
        assert (ink & off_arm.astype(np.uint8)).sum() == 0  # none of the wall inside the selection

    def test_ink_is_only_inside_the_selection(self):
        img, sel, _, _ = _arm_scene()
        tight = np.zeros_like(sel)
        tight[100:140, 140:200] = 1  # covers only part of the tattoo
        ink, _, _ = T.find_ink(img, tight, 0.3)
        assert ink[tight == 0].sum() == 0

    def test_the_skin_colour_ignores_the_background_in_the_selection(self):
        img, sel, _, _ = _arm_scene()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        roi = cv2.dilate(sel, np.ones((3, 3), np.uint8), iterations=20) > 0
        med, _ = T.skin_colour(lab, T.skin_prior(img), roi)
        skin_lab = cv2.cvtColor(np.uint8([[SKIN * 0.93]]), cv2.COLOR_BGR2LAB).astype(np.float32)[
            0, 0
        ]
        wall_lab = cv2.cvtColor(np.uint8([[WALL]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        assert np.linalg.norm(med - skin_lab) < np.linalg.norm(med - wall_lab) / 3

    def test_more_grow_catches_at_least_as_much(self):
        img, sel, _, _ = _arm_scene()
        low, _, _ = T.find_ink(img, sel, 0.0)
        high, _, _ = T.find_ink(img, sel, 1.0)
        assert high.sum() >= low.sum()

    def test_clean_skin_gives_no_ink(self):
        img, _, _, _ = _arm_scene()
        sel = np.zeros(img.shape[:2], np.uint8)
        sel[85:105, 20:80] = 1  # a patch of plain arm, far from the tattoo
        ink, _, _ = T.find_ink(img, sel, 0.3)
        assert ink.sum() == 0


class TestRemoveTattoo:
    def test_tattoo_becomes_skin(self):
        img, sel, inked, _ = _arm_scene()
        out = T.remove_tattoo(img, sel, algorithm="telea")
        m = inked > 0
        assert _lab_dist(img[m].mean(0), SKIN * 0.93) > 40  # clearly inked before
        assert _lab_dist(out[m].mean(0), SKIN * 0.93) < 12  # skin-coloured after

    def test_the_background_inside_the_selection_is_untouched_bit_for_bit(self):
        # The bug this guards against: painting past the arm pulled wall colour into the fill.
        img, sel, _, arm = _arm_scene()
        out = T.remove_tattoo(img, sel, algorithm="telea")
        wall = (arm == 0) & (sel > 0)
        assert np.array_equal(out[wall], img[wall])

    def test_no_wall_colour_bleeds_into_the_arm(self):
        img, sel, inked, _ = _arm_scene()
        out = T.remove_tattoo(img, sel, algorithm="telea")
        ring = (cv2.dilate(inked, np.ones((3, 3), np.uint8), iterations=6) > 0) & (inked == 0)
        # The blue wall has a very different hue; a bleed would drag the fill toward it.
        m = inked > 0
        assert _lab_dist(out[m].mean(0), WALL) > _lab_dist(out[m].mean(0), SKIN * 0.93) * 3
        assert ring.any()

    def test_pixels_outside_the_selection_are_untouched(self):
        img, sel, _, _ = _arm_scene()
        out = T.remove_tattoo(img, sel, algorithm="telea")
        assert np.array_equal(out[sel == 0], img[sel == 0])

    def test_a_tight_selection_on_the_edge_of_the_arm_does_not_pull_in_the_wall(self):
        img, _, _, arm = _arm_scene()
        sel = np.zeros(img.shape[:2], np.uint8)
        sel[55:110, 150:215] = 1  # runs off the top of the arm onto the wall
        out = T.remove_tattoo(img, sel, algorithm="telea")
        wall = (arm == 0) & (sel > 0)
        assert np.array_equal(out[wall], img[wall])

    def test_fill_has_skin_like_grain(self):
        img, sel, inked, _ = _arm_scene()
        out = T.remove_tattoo(img, sel, algorithm="telea")

        def grain(a, m):
            l_ = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]
            return float((l_ - cv2.GaussianBlur(l_, (0, 0), 1.5))[m].std())

        inner = cv2.erode(inked, np.ones((3, 3), np.uint8), iterations=4) > 0
        plain = np.zeros_like(inked, bool)
        plain[85:105, 20:80] = True
        assert grain(out, inner) > 0.5 * grain(img, plain)  # a smooth fill would be near 0

    def test_strength_zero_is_a_no_op_and_half_is_in_between(self):
        img, sel, inked, _ = _arm_scene()
        assert np.array_equal(T.remove_tattoo(img, sel, strength=0.0, algorithm="telea"), img)
        full, half = (
            T.remove_tattoo(img, sel, strength=1.0, algorithm="telea"),
            T.remove_tattoo(img, sel, strength=0.5, algorithm="telea"),
        )
        m = inked > 0
        d = lambda a: _lab_dist(a[m].mean(0), SKIN * 0.93)  # noqa: E731
        assert d(full) < d(half) < d(img)

    def test_deterministic(self):
        img, sel, _, _ = _arm_scene()
        assert np.array_equal(
            T.remove_tattoo(img, sel, algorithm="telea"),
            T.remove_tattoo(img, sel, algorithm="telea"),
        )

    def test_nothing_to_remove_returns_a_copy(self):
        img, _, _, _ = _arm_scene()
        sel = np.zeros(img.shape[:2], np.uint8)
        sel[85:105, 20:80] = 1
        out = T.remove_tattoo(img, sel, algorithm="telea")
        assert np.array_equal(out, img) and out is not img
        empty = T.remove_tattoo(img, np.zeros(img.shape[:2], np.uint8), algorithm="telea")
        assert np.array_equal(empty, img) and empty is not img

    def test_a_huge_tattoo_still_gets_removed_and_says_how(self):
        # A "sleeve": the ink fills the whole selection, so the careful detector has no skin around it to go by.
        img, _, _, _ = _arm_scene()
        sel = np.zeros(img.shape[:2], np.uint8)
        sel[95:145, 130:230] = 1
        sleeve = img.copy()
        sleeve[95:145, 130:230] = (INK + np.random.default_rng(2).normal(0, 4, (50, 100, 3))).clip(
            0, 255
        )
        out, info = T.remove_tattoo_info(sleeve, sel, algorithm="telea")
        assert info["mode"] in ("detected", "relaxed", "selection") and info["ink_pct"] > 30
        assert not np.array_equal(out, sleeve)  # it never silently does nothing
        assert (
            _lab_dist(out[100:140, 140:220].reshape(-1, 3).mean(0), INK) > 25
        )  # and the ink is gone

    def test_plain_skin_selection_is_reported_as_clean_and_left_alone(self):
        img, _, _, _ = _arm_scene()
        sel = np.zeros(img.shape[:2], np.uint8)
        sel[85:105, 20:80] = 1
        out, info = T.remove_tattoo_info(img, sel, algorithm="telea")
        assert info["mode"] == "clean" and np.array_equal(out, img)

    def test_info_for_a_normal_tattoo(self):
        img, sel, _, _ = _arm_scene()
        _, info = T.remove_tattoo_info(img, sel, algorithm="telea")
        assert info["mode"] == "detected" and 0 < info["ink_pct"] < 60

    def test_bad_inputs(self):
        img, sel, _, _ = _arm_scene()
        with pytest.raises(ValidationError):
            T.remove_tattoo(img, sel.astype(np.float32), algorithm="telea")
        with pytest.raises(ValidationError):
            T.remove_tattoo(img, sel[:10], algorithm="telea")

    def test_alpha_channel_is_dropped_not_a_crash(self):
        img, sel, _, _ = _arm_scene()
        bgra = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        assert T.remove_tattoo(bgra, sel, algorithm="telea").shape == img.shape

    def test_other_ink_colours_and_skin_tones(self):
        # A dark-skinned arm with a red tattoo: the skin prior and the colour distance still separate them.
        img, sel, inked, _ = _arm_scene()
        dark = np.array([40, 70, 110], np.float32)
        arm_rows = np.zeros(img.shape[:2], bool)
        arm_rows[70:170] = True
        recol = img.astype(np.float32)
        recol[arm_rows & (inked == 0)] = dark * (0.9 + 0.1 * np.random.default_rng(1).random((1,)))
        recol[inked > 0] = (30, 30, 200)  # red ink
        recol = np.clip(recol, 0, 255).astype(np.uint8)
        out = T.remove_tattoo(recol, sel, algorithm="telea")
        m = inked > 0
        assert _lab_dist(out[m].mean(0), dark) < 15


class TestEndpoint:
    @staticmethod
    def _post(client, img, mask, **data):
        files = {
            "file": ("a.png", cv2.imencode(".png", img)[1].tobytes(), "image/png"),
            "mask": ("m.png", cv2.imencode(".png", mask * 255)[1].tobytes(), "image/png"),
        }
        return client.post("/api/v1/tattoo", files=files, data={"algorithm": "telea", **data})

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
