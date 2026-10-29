#!/bin/bash
# Build MoonEP (submodules/MoonEP, the OpenEuroLLM fork) for MareNostrum 5: the CUDA extension
# moonep._C is compiled in the training image (its torch / CUDA) on the login node, into a
# directory versioned by the submodule commit, $R/pylibs_moonep_<sha>. Not into the synced
# snapshot: the next sync (rsync --delete) would remove a cluster-built .so. Configs put the
# directory first on PYTHONPATH (it replaces pylibs_moonep2 = the patched d86fda4 install).
#
#   run on the MN5 login node from a synced snapshot:
#   cd $R/autoexp_<tag> && scripts/korbi/build_moonep.sh            # -> $R/pylibs_moonep_<sha>
set -euo pipefail
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
SIF=${SIF:-$R/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif}
cd "$(dirname "$0")/../.."
SRC=$PWD/submodules/MoonEP
# the snapshot has no .git: the commit comes from the sync stamp, else from git
SHA=$(grep -m1 "submodules/MoonEP" SYNC_STAMP 2>/dev/null | awk '{print $3}')
SHA=${SHA:-$(git -C "$SRC" rev-parse --short HEAD)}
DEST=${1:-$R/pylibs_moonep_$SHA}
if [ -f "$DEST/moonep/__init__.py" ] && ls "$DEST"/moonep/_C*.so >/dev/null 2>&1; then
    echo "already built: $DEST"; exit 0
fi
module load SINGULARITY >/dev/null 2>&1 || true
TMP=$(mktemp -d "$R/.moonep_build_XXXX")
cp -r "$SRC"/. "$TMP"/
# H100 only (sm_90); no GPU on the login node, so the arch list must be explicit
singularity exec --bind /gpfs "$SIF" bash -c "
    cd $TMP && TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 \
    pip install --no-deps --no-build-isolation --no-index --target $DEST.tmp . > $TMP/build.log 2>&1" \
    || { tail -30 "$TMP/build.log"; exit 1; }
rm -rf "$DEST"; mv "$DEST.tmp" "$DEST"; rm -rf "$TMP"
echo "$SHA" > "$DEST/MOONEP_COMMIT"
ls "$DEST"/moonep/_C*.so
echo "built: $DEST (MoonEP $SHA)"
