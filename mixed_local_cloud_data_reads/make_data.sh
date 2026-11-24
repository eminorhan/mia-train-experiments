#!/usr/bin/env bash
# mixed_local_cloud_data_reads: write the data configs.
#
#   bash mixed_local_cloud_data_reads/make_data.sh            # all three files
#   bash mixed_local_cloud_data_reads/make_data.sh optic      # data/v3_plus_optic_lobe_{local,bucket}.yaml (arms 1a, 1b)
#   bash mixed_local_cloud_data_reads/make_data.sh buckets    # data/v3_plus_buckets.yaml (arm 1c)
#
# Each file is lmd-configs' lmd_ssl_v3 default corpus config (247 lmd volumes) under a header, minus one volume
# whose store is gone, with `zarr_version` written as `format`, plus the retry budget for bucket reads
# (`tensorstore_context`) and volumes appended at the end:
# - arms 1a and 1b: the FlyEM optic-lobe EM, read from /nrs (local) or from gs://flyem-optic-lobe (bucket). The two
#   files differ only in where that volume is read from.
# - arm 1c: six public EM volumes, each read from its bucket.
# Settings: README.md, "Data".
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC=/groups/miaai/miaai/lmd-v0.0.1/configs/configs/lmd_ssl_v3/unlabeled/8nm_p256_swe0.4_min1e8_up8_win1-99_census20261101.yaml
SRC_COMMIT=1009c6d   # lmd-configs commit of SRC
# lmd-configs has since removed SRC from its folder. Its history still has it, so read a copy:
#   git -C /groups/miaai/miaai/lmd-v0.0.1/configs show $SRC_COMMIT:configs/lmd_ssl_v3/unlabeled/<name of SRC> > copy.yaml
#   SRC_FILE=copy.yaml bash make_data.sh
SRC_FILE=${SRC_FILE:-$SRC}
[[ -f $SRC_FILE ]] || { echo "make_data.sh: no such file: $SRC_FILE (see the comment above SRC_FILE)" >&2; exit 1; }
# Left out: this store was renamed to crop-001.zarr on 2026-11-13, after the census SRC was built on, so its path
# no longer exists and miao cannot open it.
GONE='exm-mouse-liconn-RID16-retina-spinningdisk/crop-001_RID16_18x'
# lmd-configs' file names each store's format with `zarr_version`. miao calls that key `format` since v0.5.0.
corpus () {
  awk -v gone="- name: $GONE" '$0 == gone {skip = 1; next} /^- name: / {skip = 0} !skip' "$SRC_FILE" |
    sed 's/^  zarr_version: /  format: /'
}
BOX='[[6784, 29952], [18432, 49536], [25600, 43008]]'   # the optic lobe's 5% tissue box, x y z in level-0 voxels
WINDOW_MIN=${WINDOW_MIN:-0}      # p1 and p99 of the optic-lobe EM inside the box (probes/config_check on /nrs)
WINDOW_MAX=${WINDOW_MAX:-253}
WEIGHT=${WEIGHT:-20.15}          # arms 1a, 1b: the optic lobe supplies half of the windows (4.7% at weight 1)

header () {  # header <first line> <what was appended> <volume count sentence>
  cat <<EOF
# $1
# Written by make_data.sh. Everything below this header is lmd-configs' file (commit $SRC_COMMIT),
#   $SRC
# with four changes:
# - one volume left out, $GONE:
#   its store was renamed on 2026-11-13, after that file's census, and its path no longer exists;
# - $2 appended at the end (see the comment there);
# - \`tensorstore_context\` added right below this header;
# - every volume's \`zarr_version\` key written as \`format\`, the key's name in miao since v0.5.0.
# $3
# That header is lmd-configs' own.
EOF
}

context () {  # the retry budget for bucket reads (README.md, "Data"); it changes nothing for local volumes
  cat <<EOF
# Added by make_data.sh: tensorstore retries a failed bucket request 680 times, about 6 hours, before the read
# fails and ends the run. Its default is 32 retries, about 15 minutes.
tensorstore_context:
  gcs_request_retries: {max_retries: 680}     # gs:// volumes
  http_request_retries: {max_retries: 680}    # https:// volumes
#
EOF
}

write () {  # write <local|bucket> <path> <image_key> <one line saying where it is read from>
  local out="$HERE/data/v3_plus_optic_lobe_$1.yaml"
  mkdir -p "$HERE/data"
  {
    header "mixed_local_cloud_data_reads, arm \"$1\": the lmd_ssl_v3 corpus plus the FlyEM optic-lobe EM, $4." \
           "one volume" \
           "The corpus is therefore 247 volumes: 246 of the 247 that lmd-configs' header below counts, plus the optic lobe."
    cat <<EOF
# The local and bucket files differ only in the appended volume's \`path\` and \`image_key\`.
#
EOF
    context
    corpus
    cat <<EOF
# Appended by make_data.sh, not by lmd-configs: the FlyEM optic-lobe EM (Neuroglancer precomputed, uint8, 8 nm,
# x y z 36864 x 55296 x 45056), $4. The same volume is on /nrs and in the bucket.
# bounding_box: the 5% tissue box of /nrs/scicompsoft/orhane/probes/large_volumes/tissue_boxes/out/optic_lobe.json.
# normalize_min/max: p1 and p99 inside that box. weight: sets this volume's share of the windows (README.md).
- name: em-drosophila-flyem-optic-lobe/full
  path: $2
  image_key: $3
  format: precomputed
  bounding_box: $BOX
  normalize_max: $WINDOW_MAX
  normalize_min: $WINDOW_MIN
  weight: $WEIGHT
EOF
  } > "$out"
  echo "wrote $out"
}

entry () {  # entry <name> <path> <image_key> <bounding box> <normalize_min> <normalize_max>
  cat <<EOF
- name: $1
  path: $2
  image_key: $3
  format: precomputed
  bounding_box: $4
  normalize_max: $6
  normalize_min: $5
EOF
}

write_buckets () {
  local out="$HERE/data/v3_plus_buckets.yaml"
  mkdir -p "$HERE/data"
  {
    header "mixed_local_cloud_data_reads, arm 1c: the lmd_ssl_v3 corpus plus six public EM volumes, each read from its bucket." \
           "six volumes" \
           "The corpus is therefore 252 volumes: 246 of the 247 that lmd-configs' header below counts, plus the six."
    echo "#"
    context
    corpus
    cat <<EOF
# Appended by make_data.sh, not by lmd-configs: six public EM volumes, each read straight from its bucket. All are
# Neuroglancer precomputed, uint8. Each has the standard weight, so its share of the windows follows from its size
# like any lmd volume's. bounding_box: a tissue box, x y z in level-0 voxels. normalize_min/max: p1 and p99
# inside the box, measured as lmd-configs measures its windows. Sources of both: README.md, "Data".
# Optic lobe, 8 nm. The same entry as in arm 1b, at the standard weight.
EOF
    entry em-drosophila-flyem-optic-lobe/full gs://flyem-optic-lobe grayscale-clahe-jpeg "$BOX" "$WINDOW_MIN" "$WINDOW_MAX"
    echo "# Male CNS, 8 nm, the CLAHE JPEG copy. Its box overlaps the optic lobe's, which is a corner of the same tissue."
    entry em-drosophila-flyem-male-cns/full gs://flyem-male-cns em/em-clahe-jpeg \
          '[[12032, 86272], [12544, 44800], [19712, 39424]]' 0 253
    echo "# Hemibrain, 8 nm, the raw JPEG copy."
    entry em-drosophila-flyem-hemibrain/full gs://neuroglancer-janelia-flyem-hemibrain emdata/raw/jpeg \
          '[[59, 34427], [8256, 35776], [8064, 27840]]' 58 232
    echo "# H01, 4 x 4 x 33 nm at level 0. A run at 8 nm reads its 8 x 8 x 33 nm level."
    entry em-human-h01/full gs://h01-release/data/20210601 4nm_raw \
          '[[638976, 671744], [284672, 317440], [1664, 3712]]' 0 255
    echo "# MICrONS minnie65, 8 x 8 x 40 nm, the copy on Amazon (the same voxels as the copy on Google). The box is"
    echo "# counted from the volume's first voxel, which is voxel [13824, 13824, 14816] of the dataset."
    entry em-mouse-microns-minnie65/full https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65 em \
          '[[40960, 57344], [49152, 65536], [5472, 7520]]' 111 154
    echo "# MANC, 8 nm, the CLAHE JPEG copy."
    entry em-drosophila-flyem-manc/full gs://flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy jpeg \
          '[[9216, 36864], [21760, 40960], [23296, 59136]]' 0 253
  } > "$out"
  echo "wrote $out"
}

case "${1:-all}" in
  optic|all)
    write local /nrs/flyem/optic-lobe optic-lobe-grayscale "read from /nrs"
    write bucket gs://flyem-optic-lobe grayscale-clahe-jpeg "read from the public bucket gs://flyem-optic-lobe"
    ;;&
  buckets|all) write_buckets ;;&
  optic|buckets|all) ;;
  *) echo "usage: make_data.sh [optic|buckets|all]" >&2; exit 2 ;;
esac
