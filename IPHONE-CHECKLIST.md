# iPhone 14 Pro checklist

Things headless Chromium can't prove. Run through this on the phone after each release and note what breaks.
Open the app at the server address (plain `http://` over Tailscale is fine; there is no offline service worker on http).

## Before you start

- [ ] Version shown in the footer / `GET /health` matches the build (clear Safari website data once if it is old; the
      server now sends `Cache-Control: no-cache`, so later rebuilds should arrive on their own).
- [ ] iOS version noted here: ______ (decides whether WebGPU is ever a candidate for Phase 4).

## Loading photos

- [ ] A camera-roll **HEIC** photo loads and previews (Safari decodes it natively).
- [ ] A portrait photo shows the right way up (EXIF rotation), in the preview and in the result.
- [ ] A 12 MP photo loads without the tab reloading; note the time to first preview: ______ s.

## Live preview (WebGL)

- [ ] "Live preview" badge appears; sliders for grain, brightness, contrast, saturation, sharpness, vignette redraw smoothly.
- [ ] Sepia / grayscale match what Process returns (compare the preview with the Result tab).
- [ ] Rotate/flip keeps the preview in sync.
- [ ] No black or flickering canvas after switching apps and coming back (context loss recovery).

## Background

- [ ] Enable Background: status says "Finding the subject…", then the preview updates (U2NetP: a second or so).
- [ ] Blur strength slider is live; Remove shows the checkerboard; Colour swatches and the colour picker are live.
- [ ] Photo mode: choosing a photo from the camera roll works; zoom / move sliders are live; Soften edge and Edge in/out look right.
- [ ] Process gives the same picture as the preview and is quick (the matte is reused, the model runs once).

## Face swap

- [ ] Opening the Face swap group draws a ring on each face; tapping a ring selects it (rings are at least thumb-sized).
- [ ] With several faces, the numbered chips and rings stay in sync.
- [ ] Choosing a donor photo (camera roll, HEIC) works and its face count appears.
- [ ] Process shows "Swapping the face…", then a result in about 10 s; the new face sits on the right person.
- [ ] Error messages are readable: donor with no face, photo with no face, models missing.

## Layout and touch

- [ ] Nothing hides behind the notch / Dynamic Island or the home bar (safe areas); bottom nav is reachable.
- [ ] Inputs don't zoom the page when focused (16 px minimum font).
- [ ] Every button is easy to hit with a thumb (44 px minimum); sliders can be dragged without scrolling the page.
- [ ] Add to Home Screen opens full screen and the app still works.

## Notes from the last run

(date, iOS version, what failed)
