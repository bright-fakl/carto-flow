# Logo and Branding

The logo files are in `docs/assets/logo/`.

## The mark

The mark is an irregular, country-like outline cut into four regions by two wavy gaps that cross at 72 degrees. The background shows through the gaps, so the mark works on any surface. The two right-hand regions are amber and the two left-hand regions are blue.

<div style="display:flex;gap:1rem;flex-wrap:wrap">
  <div style="background:#ffffff;padding:1rem;border:1px solid #8884"><img src="../assets/logo/wordmark-light.svg" alt="carto-flow wordmark, light variant" height="64"></div>
  <div style="background:#1b1f24;padding:1rem"><img src="../assets/logo/wordmark-dark.svg" alt="carto-flow wordmark, dark variant" height="64"></div>
</div>

## Files

| File | Use |
|---|---|
| `mark-light.svg`, `mark-dark.svg` | Mark for light and dark surfaces (gap width 8 in a 96-unit mark) |
| `mark-hero-light.svg`, `mark-hero-dark.svg` | Mark for sizes above 256 px (gap width 6) |
| `wordmark-light.svg`, `wordmark-dark.svg` | Mark and the name `carto-flow`, text as outlined paths |
| `mark-mono.svg` | One color; the fill is `currentColor` and renders black when used as an image |
| `favicon.svg` | Mark with gap width 10 for 16 px; switches colors with `prefers-color-scheme` |
| `favicon-32.png`, `apple-touch-icon-180.png` | Raster icons (the touch icon has an opaque white background) |
| `social-preview-1280x640.png` | Repository social preview |

## Colors

| Variant | Blue | Amber | Text | Use on |
|---|---|---|---|---|
| Light | `#2c6e9e` | `#c27400` | `#1b1f24` | white and light surfaces |
| Dark | `#85b7e0` | `#facc38` | `#e8edf2` | dark surfaces, e.g. `#1e2029` (docs, slate scheme) and `#000000` (docs header) |

The blue and the dark amber are the `PlotStyle` colors. The light amber is darker than the `PlotStyle` highlight (`#d98200`, 2.9:1 on white) because WCAG 1.4.11 requires 3:1 for graphical objects.

WCAG contrast ratios against the docs surfaces:

| Color | On `#ffffff` | On `#1e2029` | On `#000000` |
|---|---|---|---|
| Light blue `#2c6e9e` | 5.48 | not used | not used |
| Light amber `#c27400` | 3.63 | not used | not used |
| Light text `#1b1f24` | 16.56 | not used | not used |
| Dark blue `#85b7e0` | not used | 7.54 | 9.85 |
| Dark amber `#facc38` | not used | 10.53 | 13.76 |
| Dark text `#e8edf2` | not used | 13.65 | 17.83 |

## Size and clear space

- Keep a clear space of 25% of the mark height on every side.
- The mark is legible down to 32 px high. Below that, down to 16 px, use `favicon.svg`, which has wider gaps.
- Use the wordmark at 32 px high or more.

## Do and do not

- Use the light variant on light surfaces and the dark variant on dark surfaces.
- Do not recolor the mark or the wordmark, except for `mark-mono.svg`.
- Do not stretch or rotate the files, and do not change the gap width.
- Do not put the light variant on a dark surface or the dark variant on a light surface.
- Do not retype the wordmark in another font; use the files.

## License

The mark and the wordmark layout are original work, released under the project's [MIT license](https://github.com/bright-fakl/carto-flow/blob/main/LICENSE). The letters of the wordmark are outlines of DejaVu Sans Bold (the tagline in the social preview is DejaVu Sans). DejaVu fonts are distributed under a free license that permits use, embedding and redistribution; see [dejavu-fonts.github.io/License.html](https://dejavu-fonts.github.io/License.html).

## Regenerating

`scripts/make_logo_assets.py` generates every file in `docs/assets/logo/`:

```bash
uv run --with cairosvg --with pillow --with fonttools python scripts/make_logo_assets.py
```
