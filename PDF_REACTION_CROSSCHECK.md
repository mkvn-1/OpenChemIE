# PDF Reaction Cross-Check

PDF checked: `example/acs.joc.2c00749.pdf`

Audit output checked locally: `example/gpu_test_output/script_reaction_audit_strict`

## Result

All visible article reaction figures found during the direct page review are represented in the extraction overlays.

The strict audit extracted 14 figure/panel overlays with 105 reactions total. The audit also reported no captioned figure pages without extraction.

## Evidence

| PDF page | Visible reaction content in PDF | Extraction overlay coverage | Reaction count |
| --- | --- | --- | --- |
| 1 | Title-page chemistry graphic | `overlay_01_page_01_src_01.png` | 6 |
| 2 | Figure 1, Figure 2 | `overlay_02_page_02_src_02.png`, `overlay_03_page_02_src_03.png` | 22 |
| 3 | Figure 3, Figure 4 | `overlay_04_page_03_src_05.png`, `overlay_05_page_03_src_06.png` | 11 |
| 4 | Figure 5, Figure 6 | `overlay_06_page_04_src_08.png`, `overlay_07_page_04_src_09.png` | 14 |
| 5 | Figure 7, Figure 8 | `overlay_08_page_05_src_10.png`, `overlay_09_page_05_src_11.png` | 11 |
| 6 | Figure 9, Figure 10 | `overlay_10_page_06_src_12.png`, `overlay_11_page_06_src_14.png` | 15 |
| 7 | Figure 11 | `overlay_12_page_07_src_15_left.png`, `overlay_13_page_07_src_15_right.png` | 18 |
| 8 | Figure 12 | `overlay_14_page_08_src_16.png` | 8 |
| 9 | References only | No extraction expected | 0 |
| 10 | References and ACS recommendations only | Zero-reaction false positive dropped | 0 |

## Important Checks

- Caption pages from PDF text: pages 2-8 contain Figure 1 through Figure 12.
- `caption_pages_without_extractions` was empty.
- Figure 11 was the previous miss. It is now handled by panel fallback using vertical halves: left panel 10 reactions, right panel 8 reactions.
- Three nested duplicate crops were dropped because their contents were already contained inside kept figures.
- One page-10 false positive was dropped because it contained zero reactions.

## Caveat

This verifies visible figure-region recall against the rendered PDF pages and extraction overlays. It does not prove atom-level or molecule-level chemical correctness for every extracted reaction without a manually labeled chemistry ground truth set.
