# Demo media

Drop generated clips here to populate the example grid in the top-level
README. Media files under `assets/` are allowed through `.gitignore`
(everything else that is `*.mp4` / `*.gif` / `*.png` stays ignored) and are
stored with Git LFS per `.gitattributes`.

Suggested names, matching the commented-out grid in `README.md`:

| file | content |
| --- | --- |
| `t2v_swimmer.gif` | text-to-video: the swimmer prompt in `inference/prompts/examples.json` |
| `t2v_showroom.gif` | text-to-video: the concept-car showroom prompt |
| `t2v_scooter.gif` | text-to-video: the 2D-animated scooter prompt |
| `i2v_example.gif` | image-to-video from a first frame |

Keep GIFs under ~5 MB each so the README loads quickly; a 480p, 24 fps, 4 s
excerpt is plenty.
