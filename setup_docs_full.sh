#!/bin/bash
# setup_docs_full.sh — Copy audio files and deploy the full landing page
# Run from the repo root: bash setup_docs_full.sh

set -e

SRC="AUDIO EVAL TRIM/ALL"
DST="docs-full/assets/audio/samples"

echo "=== Creating audio directory ==="
mkdir -p "$DST"

echo "=== Copying audio files ==="
# DiffMean pitch
cp "$SRC/DM_pitch_low_to_high_song1109_baseline.mp3" "$DST/"
cp "$SRC/DM_pitch_low_to_high_song1109_steered_a+0.1.mp3" "$DST/"
cp "$SRC/DM_pitch_high_to_low_song708_baseline.mp3" "$DST/"
cp "$SRC/DM_pitch_high_to_low_song708_steered_a-0.5.mp3" "$DST/"
cp "$SRC/DM_pitch_high_to_low_song389_baseline.mp3" "$DST/"
cp "$SRC/DM_pitch_high_to_low_song389_steered_a-0.1.mp3" "$DST/"

# DiffMean duration
cp "$SRC/DM_duration_low_to_high_Kunstderfuge-766_baseline.mp3" "$DST/"
cp "$SRC/DM_duration_low_to_high_Kunstderfuge-766_steered_a+0.5.mp3" "$DST/"
cp "$SRC/DM_duration_high_to_low_Musicalion-1698_baseline.mp3" "$DST/"
cp "$SRC/DM_duration_high_to_low_Musicalion-1698_steered_a-0.5.mp3" "$DST/"

# DiffMean dual
cp "$SRC/DM_dual_low_short_to_high_long_Kunstderfuge-1367_baseline.mp3" "$DST/"
cp "$SRC/DM_dual_low_short_to_high_long_Kunstderfuge-1367_steered_ap+1.5_ad+0.5.mp3" "$DST/"
cp "$SRC/DM_dual_high_long_to_low_long_Musicalion-3672_baseline.mp3" "$DST/"  
cp "$SRC/DM_dual_high_long_to_low_long_Musicalion-3672_steered_ap-1.5_ad-1.2.mp3" "$DST/"

# SAS pitch
cp "$SRC/SAS_pitch_low_to_high_Musicalion-3512_baseline.mp3" "$DST/"
cp "$SRC/SAS_pitch_low_to_high_Musicalion-3512_steered_l+0.75.mp3" "$DST/"
cp "$SRC/SAS_pitch_high_to_low_Kunstderfuge-389_baseline.mp3" "$DST/"
cp "$SRC/SAS_pitch_high_to_low_Kunstderfuge-389_steered_l+0.5.mp3" "$DST/"

# SAS duration
cp "$SRC/SAS_duration_short_to_long_Musicalion-962_baseline.mp3" "$DST/"
cp "$SRC/SAS_duration_short_to_long_Musicalion-962_steered_l+1.0.mp3" "$DST/"
cp "$SRC/SAS_duration_long_to_short_Musicalion-1416_baseline.mp3" "$DST/"
cp "$SRC/SAS_duration_long_to_short_Musicalion-1416_steered_l-1.5.mp3" "$DST/"

# SAS dual
cp "$SRC/SAS_dual_low_to_high_long_Musicalion-1109_baseline.mp3" "$DST/"
cp "$SRC/SAS_dual_low_to_high_long_Musicalion-1109_steered_lp+0.50_ld+0.75_deg0.07.mp3" "$DST/"
cp "$SRC/SAS_dual_high_to_low_short_Musicalion-708_baseline.mp3" "$DST/"
cp "$SRC/SAS_dual_high_to_low_short_Musicalion-708_steered_lp-0.50_ld-0.25_deg1.00.mp3" "$DST/"

# Smooth steering
cp "$SRC/SMOOTH_pitch_low_to_high_song353_abrupt.mp3" "$DST/"
cp "$SRC/SMOOTH_pitch_low_to_high_song353_warmup_hold_32beats.mp3" "$DST/"
cp "$SRC/SMOOTH_pitch_high_to_low_song389_abrupt_a-1.mp3" "$DST/"
cp "$SRC/SMOOTH_pitch_high_to_low_song389_gradual_256beats.mp3" "$DST/"
cp "$SRC/SMOOTH_duration_short_to_long_song766_abrupt.mp3" "$DST/"
cp "$SRC/SMOOTH_duration_short_to_long_song766_gradual_128beats.mp3" "$DST/"
cp "$SRC/SMOOTH_duration_long_to_short_song3708_abrupt.mp3" "$DST/"
cp "$SRC/SMOOTH_duration_long_to_short_song3708_gradual_128beats.mp3" "$DST/"

echo ""
echo "=== Done! $(ls "$DST" | wc -l | tr -d ' ') audio files copied ==="
echo ""
echo "Next steps:"
echo "  1. Create a NEW repo on GitHub: music-transformer-sae-full"
echo "     (or push docs-full/ as a subtree to the existing repo)"
echo ""
echo "  Option A — New standalone repo:"
echo "    cd docs-full"
echo "    git init"
echo "    git add ."
echo '    git commit -m "Landing page with DiffMean + SAS audio examples"'
echo "    git remote add origin git@github.com:GiannisProkopiouOrfium/music-transformer-sae-full.git"
echo "    git push -u origin main"
echo "    # Then: Settings → Pages → Source: main / root → Save"
echo ""
echo "  Option B — Same repo, different folder (GitHub Pages project page):"
echo "    # Push this branch with docs-full/ committed"
echo "    # Settings → Pages → Source: feat/sparse-steering / docs-full"
echo "    # URL: https://giannisprokopiouorfium.github.io/music-transformer-sae/full/"
echo ""
echo "  Option C — Same repo, replace docs/ on a new branch:"
echo "    git checkout -b gh-pages-full"
echo "    rm -rf docs"
echo "    mv docs-full docs"
echo '    git add . && git commit -m "Full landing page with both methods"'
echo "    git push origin gh-pages-full"
echo "    # Settings → Pages → Source: gh-pages-full / docs"
