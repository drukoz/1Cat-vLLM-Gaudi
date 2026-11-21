#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# layout_2xtp4.sh SHA12 [ARM...]: the layout re-check of the integration build (coordinator, build 4): two TP4 + EP4
# servers side by side in ONE all-card queue job, modules 0-3 (cards 0,1,2,7) and 4-7 (cards 3,4,5,6), as coord's
# 00048 did before the QSA change and MTP. Stage-1 form (CARD.md, "A job that starts its own containers"): the job
# runs on the node host and starts one container per module group itself:
#
#   printf 'bash $HOME/farm/integration/results/late/layout_2xtp4.sh SHA12\n' | bin/farm submit integration layout-2xtp4 all 50m
#
# Each container: the image farm-integration:SHA12 with the bundle ~/farm/integration/b/SHA12 at /bundle (read-only),
# its cards by index (HABANA_VISIBLE_DEVICES, never HABANA_VISIBLE_MODULES), its module group's CPUs, the queue's
# label, its own ports (the job's base and base + 10). It runs the kit's headline_job.sh with the ARMs (default
# mtp3 mtp1) at 16 streams, two measured rounds, greedy then sampled (VLLM_HPU_SPEC_SAMPLED_VERIFY=2), and build 4's
# server env (the farm preset and the build's flags come from headline_job.sh; the prefix flags, the MoE chunk size
# and the synchronous accumulation-queue flushing from here). The two containers start together; their rounds overlap
# but are not barrier-aligned. Results: ~/farm/integration/results/<job>-<name>/{tp4a,tp4b}/ and the two logs.
set -u
SHA="${1:?usage: layout_2xtp4.sh SHA12 [ARM...]}"
shift
ARMS="${*:-mtp3 mtp1}"
IMG="farm-integration:$SHA"
BUNDLE="$HOME/farm/integration/b/$SHA"
RUN="${CARDQ_JOB:-manual}-${CARDQ_NAME:-layout-2xtp4}"
R="$HOME/farm/integration/results/$RUN"
PB="${CARDQ_PORT_BASE:-8100}"
mkdir -p "$R"
[ -d "$BUNDLE/plugin" ] || { echo "layout_2xtp4: no bundle $BUNDLE/plugin" >&2; exit 2; }
sudo -n docker image inspect "$IMG" >/dev/null 2>&1 || { echo "layout_2xtp4: no image $IMG" >&2; exit 2; }
echo "layout_2xtp4: image $IMG, arms $ARMS, results $R, ports $PB and $((PB + 10)), $(date -u +%T)"

one() {  # LABEL CARD_INDICES CPUS PORT
    mkdir -p "$R/$1"
    sudo -n docker run --rm --name "integ-${CARDQ_JOB:-manual}-$1" --label "cardq.job=${CARDQ_JOB:-manual}" \
        --init --runtime=habana --cap-add=sys_nice --net=host --ipc=host \
        -e HABANA_VISIBLE_DEVICES="$2" --cpuset-cpus "$3" \
        -w /bundle/plugin -v "$BUNDLE:/bundle:ro" -v "$R:/results" -v /models:/models:ro \
        -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PYTHONUNBUFFERED=1 -e PYTHONFAULTHANDLER=1 \
        -e PYTHONDONTWRITEBYTECODE=1 -e PT_HPU_LAZY_MODE=1 -e PT_HPU_LAZY_ACC_PAR_MODE=0 \
        -e HABANA_LOGS="/results/$1/habana_logs" -e FARM_OUT="/results/$1" -e FARM_NCARDS=4 -e FARM_PORT_BASE="$4" \
        -e HEADLINE_LEVELS=16 -e HEADLINE_ROUNDS=2 -e HEADLINE_MTP_SAMPLED=1 -e HEADLINE_MTP_SAMPLED_LEVELS=16 \
        -e VLLM_HPU_SPEC_SAMPLED_VERIFY=2 -e VLLM_HPU_MTP_TIMING=0 -e HEADLINE_MTP_FALLBACK=0 -e HEADLINE_HOLD_S=3000 \
        -e VLLM_HPU_PREFIX_ALIGN_SPLIT=1 -e VLLM_HPU_PREFIX_KEEP_PROMPT_END=1 \
        -e VLLM_MOE_CHUNK=64,8192 -e VLLM_MOE_TOKEN_BOUNDARY=512,1000000 -e PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1 \
        -e HEADLINE_BARRIER_DIR=/results/barrier -e HEADLINE_BARRIER_N=2 -e HEADLINE_BARRIER_NAME="$1" \
        "$IMG" bash /bundle/plugin/tools/flash_next/integration/headline_job.sh $ARMS
}

one tp4a 0,1,2,7 0-17,19-36,38-55,57-74 "$PB" >"$R/tp4a.log" 2>&1 &
PA=$!
one tp4b 3,4,5,6 76-93,95-112,114-131,133-150 "$((PB + 10))" >"$R/tp4b.log" 2>&1 &
PBID=$!
wait "$PA"
RA=$?
wait "$PBID"
RB=$?
for s in tp4a tp4b; do
    echo "=== layout_2xtp4: $s"
    grep -E "SERVER-|^MEAS|^SAMPLED|^SPEC|arm .* rc=|failed;" "$R/$s.log" | cut -c1-240
    [ -f "$R/$s/headline.md" ] && sed -n '1,20p' "$R/$s/headline.md"
done
echo "layout_2xtp4: tp4a rc=$RA, tp4b rc=$RB, $(date -u +%T)"
exit $((RA + RB))
