#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run the Flash-Next rental test on the rented Gaudi node (RUK-2709 M4). Run it from the unpacked
# rental bundle: bash run_rental.sh --model /models/Qwen3.8-Flash-Next-FP8
#
# Starts the Habana container, runs container_entry.sh inside it (install + orchestrator), and
# leaves <results>/<run-id>.tar.gz on the host to send back. Bucket warmup is skipped by default
# (--full-warmup to force it) and the HPU recipe cache lives next to the checkpoint (the data
# disk) so compiled graphs survive between runs (--recipe-cache DIR / --no-recipe-cache).
#
# Ctrl-C, SIGTERM and a dropped SSH session (SIGHUP) stop the run's container, so no server keeps
# the cards, and still write the tarball and the two closing lines; the exit code is then 130, 143
# or 129. --download-only starts the checkpoint download in a container of its own and returns.

set -euo pipefail

DEFAULT_IMAGE="vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest"
NAME="fn-rental"
DOWNLOAD_NAME="fn-download"
# The checkpoint and the hub client of container_entry.sh (its REPO_ID and its huggingface_hub pin):
# --download-only uses them outside the run container. A test keeps the two scripts equal.
REPO_ID="Qwen/Qwen3.8-Flash-Next-FP8"
HF_HUB_VERSION="1.33.0"

# The orchestrator's phases in run order: the gate phases (GATE_PHASES in main.py), then the
# variants of the run (the ids of variants.json; --variants picks them, P runs only with
# --with-profile), then "report". --from is checked against them here, before the container starts,
# so a misspelt phase costs nothing. A test keeps these lists equal to main.py and variants.json.
GATE_PHASES="preflight build fused_gdn tiny card_tests"
KNOWN_VARIANTS="A B B4 B5 C P"
DEFAULT_VARIANTS="A B C"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "usage: $(basename "${BASH_SOURCE[0]}") --model DIR [options]"
    echo "  --model DIR       checkpoint directory (required; /models/<name> inside the container)"
    echo "  --image IMG       container image (default: $DEFAULT_IMAGE)"
    echo "  --results DIR     results directory on the host (default: ./results)"
    echo "  --variants LIST   variant ids, e.g. A,B,C (passed to the orchestrator)"
    echo "  --with-profile    enable profiling (passed to the orchestrator)"
    echo "  --download        download the checkpoint into --model inside the container first"
    echo "  --download-only   only start the checkpoint download into --model, in a container of its own"
    echo "                    ($DOWNLOAD_NAME, detached, no card), and return; tiny_smoke.sh can run meanwhile"
    echo "  --from PHASE      resume the run --run-id ID from PHASE (--run-id is required with --from)."
    echo "                    The phases, in run order: $GATE_PHASES,"
    echo "                    then the variants of the run ($KNOWN_VARIANTS; P only with --with-profile), then report"
    echo "  --run-id ID       run id (default: UTC timestamp)"
    echo "  --quick-gates     skip the recorded-only steps before the first measurement: G1b's eager"
    echo "                    replays and bench, the verify crash probe and the card_tests phase (passed"
    echo "                    to the orchestrator; the checks that decide the GDN kernel still run)"
    echo "  --no-verify-probe skip only the automatic crash probe that follows a crashed or hung GDN"
    echo "                    verify op test (passed to the orchestrator)"
    echo "  --hc-fused        ask for the fused hyper-connection kernels (off by default; passed to the"
    echo "                    orchestrator): the servers get VLLM_HPU_HC_FUSED=1 only when the card test of"
    echo "                    the kernels in gate fused_gdn passed, otherwise VLLM_HPU_HC_FUSED=0 and the run"
    echo "                    goes on with the torch hyper-connections; a tiny gate that fails with the"
    echo "                    kernels runs once more without them"
    echo "  --moe-glue        ask for the MoE glue kernels (off by default; passed to the orchestrator):"
    echo "                    the servers get VLLM_HPU_MOE_GLUE=1 only when the card test of the kernels in"
    echo "                    gate fused_gdn passed, otherwise VLLM_HPU_MOE_GLUE=0 and the run goes on with"
    echo "                    the torch MoE glue; a tiny gate that fails with the kernels runs once more"
    echo "                    without them (one retry for --hc-fused and --moe-glue together)"
    echo "  --load-read-timeout S     measured load: seconds a request may go without receiving data"
    echo "                    (passed to the orchestrator; its default: 1800)"
    echo "  --warm-request-timeout S  warm pass: seconds a request may go without receiving data, first"
    echo "                    graph compiles included (passed to the orchestrator; its default: 3600)"
    echo "  --full-warmup     run the full bucket warmup at server start (default: skipped, graphs"
    echo "                    compile on first use)"
    echo "  --server-timeout S  seconds to wait for a server to become healthy (default: 1800, enough"
    echo "                    with the default skipped warmup; raise it with --full-warmup)"
    echo "  --recipe-cache DIR  host folder for the HPU recipe cache, mounted at /recipe-cache and"
    echo "                    handed to the orchestrator, which gives it to every server when the combined"
    echo "                    kernel database is in use, and otherwise only to servers that run without"
    echo "                    the fused GDN kernel (the bridge accepts one kernel library next to a"
    echo "                    recipe cache); default: a recipe-cache folder next to --model"
    echo "  --no-recipe-cache   do not mount an HPU recipe cache"
    echo "  --no-combined-kernel-db  keep the combined kernel database out of this run"
    echo "                    (FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0 in the container): the orchestrator then"
    echo "                    treats it as not installed, so servers with the fused GDN kernel run the"
    echo "                    colon-list GC_KERNEL_PATH without a recipe cache; rerun with this option when"
    echo "                    a server fails at start with the combined database"
    echo "  --dry-run         print the docker command and exit 0"
    echo "  --no-disk-check   skip the free-space checks (download folder and Docker root)"
    echo "  --no-warm-pass    skip the warm pass that compiles the workload's graphs and caches the"
    echo "                    shared block before the coherence gate (on by default)"
}

# The nearest existing folder at or above $1 (a folder that does not exist yet lives on its parent's disk).
existing_dir() {
    local dir="$1"
    while [ ! -d "$dir" ]; do dir="$(dirname "$dir")"; done
    echo "$dir"
}

# Prints "<free bytes> <mount point>" for the filesystem holding folder $1, or nothing when df fails.
disk_free() {
    local kb="" mnt=""
    read -r kb mnt < <(df -Pk "$1" 2>/dev/null | awk 'NR==2 {print $4, $6}') || true
    case "$kb" in
        '' | *[!0-9]*) return 0 ;;
    esac
    echo "$((kb * 1024)) $mnt"
}

need_value() {
    # need_value OPTION ARGC [NEXT] — a usage error when no value follows OPTION (it is the last
    # word, or the next word is another option).
    local missing=0
    if [ "$2" -lt 2 ]; then missing=1; fi
    case "${3:-}" in
        --*) missing=1 ;;
    esac
    if [ "$missing" -eq 1 ]; then
        echo "error: $1 needs a value" >&2
        usage >&2
        exit 2
    fi
}

# True when $1 is a plain number of seconds (digits, with an optional decimal part).
is_seconds() {
    case "$1" in
        '' | *[!0-9.]* | *.*.* | .* | *.) return 1 ;;
    esac
    return 0
}

# True when the word $1 is one of the following words.
has_word() {
    local word="$1" other
    shift
    for other in "$@"; do
        if [ "$other" = "$word" ]; then return 0; fi
    done
    return 1
}

# True when the name list $1 (one per line) has the line $2.
has_name() {
    local nl=$'\n'
    case "$nl$1$nl" in
        *"$nl$2$nl"*) return 0 ;;
    esac
    return 1
}

# Prints (to stderr) the run ids under the results folder: the folders the orchestrator wrote a
# manifest.json into, oldest first for the default timestamp ids.
list_run_ids() {
    local dir found=0
    echo "run ids under $RESULTS:" >&2
    if [ -d "$RESULTS" ]; then
        for dir in "$RESULTS"/*/; do
            if [ -f "${dir}manifest.json" ]; then
                echo "  $(basename "$dir")" >&2
                found=1
            fi
        done
    fi
    if [ "$found" -eq 0 ]; then
        echo "  (none: no folder with a manifest.json there, so there is no run to resume; drop --from)" >&2
    fi
}

MODEL=""
IMAGE="$DEFAULT_IMAGE"
RESULTS=""
VARIANTS=""
WITH_PROFILE=0
DOWNLOAD=0
FROM_PHASE=""
RUN_ID=""
DRY_RUN=0
NO_DISK_CHECK=0
NO_WARM_PASS=0
FULL_WARMUP=0
SERVER_TIMEOUT=""
RECIPE_CACHE=""
NO_RECIPE_CACHE=0
NO_COMBINED_DB=0
DOWNLOAD_ONLY=0
QUICK_GATES=0
NO_VERIFY_PROBE=0
LOAD_READ_TIMEOUT=""
WARM_REQUEST_TIMEOUT=""
HC_FUSED=0
MOE_GLUE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model)
            need_value "$1" $# "${2:-}"
            MODEL="$2"
            shift 2
            ;;
        --image)
            need_value "$1" $# "${2:-}"
            IMAGE="$2"
            shift 2
            ;;
        --results)
            need_value "$1" $# "${2:-}"
            RESULTS="$2"
            shift 2
            ;;
        --variants)
            need_value "$1" $# "${2:-}"
            VARIANTS="$2"
            shift 2
            ;;
        --with-profile)
            WITH_PROFILE=1
            shift
            ;;
        --download)
            DOWNLOAD=1
            shift
            ;;
        --from)
            need_value "$1" $# "${2:-}"
            FROM_PHASE="$2"
            shift 2
            ;;
        --run-id)
            need_value "$1" $# "${2:-}"
            RUN_ID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-disk-check)
            NO_DISK_CHECK=1
            shift
            ;;
        --no-warm-pass)
            NO_WARM_PASS=1
            shift
            ;;
        --full-warmup)
            FULL_WARMUP=1
            shift
            ;;
        --server-timeout)
            need_value "$1" $# "${2:-}"
            SERVER_TIMEOUT="$2"
            if ! is_seconds "$SERVER_TIMEOUT"; then
                echo "error: --server-timeout takes a number of seconds, got '$SERVER_TIMEOUT'" >&2
                exit 2
            fi
            shift 2
            ;;
        --recipe-cache)
            need_value "$1" $# "${2:-}"  # an empty value is allowed: it means --no-recipe-cache
            RECIPE_CACHE="$2"
            if [ -z "$RECIPE_CACHE" ]; then NO_RECIPE_CACHE=1; fi
            shift 2
            ;;
        --no-recipe-cache)
            NO_RECIPE_CACHE=1
            shift
            ;;
        --no-combined-kernel-db)
            NO_COMBINED_DB=1
            shift
            ;;
        --download-only)
            DOWNLOAD_ONLY=1
            shift
            ;;
        --quick-gates)
            QUICK_GATES=1
            shift
            ;;
        --no-verify-probe)
            NO_VERIFY_PROBE=1
            shift
            ;;
        --hc-fused)
            HC_FUSED=1
            shift
            ;;
        --moe-glue)
            MOE_GLUE=1
            shift
            ;;
        --load-read-timeout)
            need_value "$1" $# "${2:-}"
            LOAD_READ_TIMEOUT="$2"
            if ! is_seconds "$LOAD_READ_TIMEOUT"; then
                echo "error: --load-read-timeout takes a number of seconds, got '$LOAD_READ_TIMEOUT'" >&2
                exit 2
            fi
            shift 2
            ;;
        --warm-request-timeout)
            need_value "$1" $# "${2:-}"
            WARM_REQUEST_TIMEOUT="$2"
            if ! is_seconds "$WARM_REQUEST_TIMEOUT"; then
                echo "error: --warm-request-timeout takes a number of seconds, got '$WARM_REQUEST_TIMEOUT'" >&2
                exit 2
            fi
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

DOWNLOAD_NEED_GB="${FN_RENTAL_DOWNLOAD_NEED_GB:-250}"
DOCKER_NEED_GB="${FN_RENTAL_DOCKER_NEED_GB:-30}"

if [ -z "$MODEL" ]; then
    echo "error: --model is required" >&2
    usage >&2
    exit 2
fi
case "$MODEL" in
    /*) ;;
    *)
        echo "error: --model must be an absolute path" >&2
        exit 2
        ;;
esac

RESULTS="${RESULTS:-$PWD/results}"

# The variants of this run, as the orchestrator selects them: --variants (default A,B,C), and P is
# added by --with-profile. An id it does not know would stop it after the install; stop here instead.
SELECTED_VARIANTS=()
VARIANT_WORDS=()
KNOWN_VARIANT_WORDS=()
read -r -a KNOWN_VARIANT_WORDS <<<"$KNOWN_VARIANTS" || true
if [ -n "$VARIANTS" ]; then
    IFS=',' read -r -a VARIANT_WORDS <<<"$VARIANTS" || true
else
    read -r -a VARIANT_WORDS <<<"$DEFAULT_VARIANTS" || true
fi
for word in ${VARIANT_WORDS[@]+"${VARIANT_WORDS[@]}"}; do
    word="${word//[[:space:]]/}"
    if [ -z "$word" ]; then continue; fi
    if ! has_word "$word" "${KNOWN_VARIANT_WORDS[@]}"; then
        echo "error: --variants: unknown variant id '$word' (the variants: $KNOWN_VARIANTS)" >&2
        exit 2
    fi
    SELECTED_VARIANTS+=("$word")
done
if [ "$WITH_PROFILE" -eq 1 ] && ! has_word P ${SELECTED_VARIANTS[@]+"${SELECTED_VARIANTS[@]}"}; then
    SELECTED_VARIANTS+=(P)
fi

# --from resumes an existing run: it needs the id of that run, and a phase of this run. Without
# these checks a missing or misspelt --run-id started an empty run under a new id, and a misspelt
# phase was found only by the orchestrator, after the container start and the install.
if [ -n "$FROM_PHASE" ]; then
    RUN_PHASES=()
    read -r -a RUN_PHASES <<<"$GATE_PHASES" || true
    RUN_PHASES+=(${SELECTED_VARIANTS[@]+"${SELECTED_VARIANTS[@]}"} report)
    if ! has_word "$FROM_PHASE" "${RUN_PHASES[@]}"; then
        echo "error: --from $FROM_PHASE is not a phase of this run; the phases, in run order: ${RUN_PHASES[*]}" >&2
        exit 2
    fi
    if [ -z "$RUN_ID" ]; then
        echo "error: --from needs --run-id ID, the id of the run to resume (without it the run would start empty under a new id)" >&2
        list_run_ids
        exit 2
    fi
    # The orchestrator resumes only a run folder that holds the manifest.json of an earlier start.
    if [ ! -f "$RESULTS/$RUN_ID/manifest.json" ]; then
        echo "error: --from $FROM_PHASE: $RESULTS/$RUN_ID has no manifest.json of an earlier start, so there is no run to resume there (check --run-id and --results, or start without --from)" >&2
        list_run_ids
        exit 2
    fi
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL_PARENT="$(dirname "$MODEL")"
MODEL_NAME="$(basename "$MODEL")"

# The recipe cache defaults to a folder next to the checkpoint, so it sits on the checkpoint's
# (data) disk and survives between runs. Relative paths are taken against $PWD.
if [ "$NO_RECIPE_CACHE" -eq 0 ] && [ -z "$RECIPE_CACHE" ]; then
    RECIPE_CACHE="$MODEL_PARENT/recipe-cache"
fi
if [ "$NO_RECIPE_CACHE" -eq 0 ]; then
    case "$RECIPE_CACHE" in
        /*) ;;
        *) RECIPE_CACHE="$PWD/$RECIPE_CACHE" ;;
    esac
fi

if [ "$NO_DISK_CHECK" -eq 1 ]; then
    echo "warning: free-space checks skipped (--no-disk-check)" >&2
elif [ "$DOWNLOAD" -eq 1 ] || [ "$DOWNLOAD_ONLY" -eq 1 ]; then
    need=$((DOWNLOAD_NEED_GB * 1000000000))
    if [ -d "$MODEL" ]; then
        used_kb="$(du -sk "$MODEL" 2>/dev/null | awk 'NR==1 {print $1}' || true)"
        case "$used_kb" in
            '' | *[!0-9]*) ;;
            *) need=$((need - used_kb * 1024)) ;;
        esac
    fi
    if [ "$need" -lt 0 ]; then need=0; fi
    dir="$(existing_dir "$MODEL")"
    free="" mnt=""
    read -r free mnt < <(disk_free "$dir") || true
    if [ -z "$free" ]; then
        echo "warning: cannot read the free space of $dir; continuing" >&2
    elif [ "$free" -lt "$need" ]; then
        echo "error: $dir (filesystem mounted at $mnt) has $((free / 1000000000)) GB free; --download needs \
$((need / 1000000000)) GB there (the 185.5 GB checkpoint plus room for the results). Fix: sudo bash \
$SCRIPT_DIR/prepare_host.sh --data-disk DEV (an unused disk from lsblk, mounted at /models), or point --model \
at a disk with room, or pass --no-disk-check." >&2
        exit 1
    else
        echo "disk: $dir has $((free / 1000000000)) GB free, --download needs $((need / 1000000000)) GB: ok"
    fi
fi
# A dry run writes nothing: the download's folder and the results folder appear only for a real run.
if [ "$DOWNLOAD" -eq 1 ] || [ "$DOWNLOAD_ONLY" -eq 1 ]; then
    if [ "$DRY_RUN" -eq 0 ]; then
        mkdir -p "$MODEL_PARENT"
    fi
elif [ ! -d "$MODEL_PARENT" ]; then
    echo "error: model parent $MODEL_PARENT does not exist (pass --download to fetch the checkpoint)" >&2
    exit 1
fi

# --download-only: the download alone, detached, in a container that holds no card and installs
# nothing from the bundle, so tiny_smoke.sh can run next to it. The free-space check above applies.
if [ "$DOWNLOAD_ONLY" -eq 1 ]; then
    DOWNLOAD_CMD=(
        docker run -d --name "$DOWNLOAD_NAME"
        -e PYTHONUNBUFFERED=1
        --net=host
        -v "$MODEL_PARENT:/models"
        "$IMAGE"
        bash -c 'pip install "huggingface_hub==$1" && hf download "$2" --local-dir "$3"'
        download "$HF_HUB_VERSION" "$REPO_ID" "/models/$MODEL_NAME"
    )
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%q ' "${DOWNLOAD_CMD[@]}"
        printf '\n'
        exit 0
    fi
    if ! command -v docker >/dev/null 2>&1; then
        echo "error: docker not found on this host; run: sudo bash $SCRIPT_DIR/prepare_host.sh" >&2
        exit 1
    fi
    if has_name "$(docker ps --format '{{.Names}}' 2>/dev/null || true)" "$DOWNLOAD_NAME"; then
        echo "error: a checkpoint download (container $DOWNLOAD_NAME) is already running; progress: sudo docker logs --tail 3 $DOWNLOAD_NAME" >&2
        exit 1
    fi
    if has_name "$(docker ps -a --format '{{.Names}}' 2>/dev/null || true)" "$DOWNLOAD_NAME"; then
        docker rm -f "$DOWNLOAD_NAME" >/dev/null  # a finished or failed earlier download; hf resumes in --model
    fi
    "${DOWNLOAD_CMD[@]}" >/dev/null
    echo "checkpoint download started in container $DOWNLOAD_NAME (detached: it goes on without this terminal)"
    echo "  progress: sudo docker logs --tail 3 $DOWNLOAD_NAME    or: du -sh $MODEL"
    echo "  finished: sudo docker wait $DOWNLOAD_NAME    (waits, then prints 0 when the checkpoint is complete)"
    echo "  then:     sudo docker rm $DOWNLOAD_NAME, and run_rental.sh --model $MODEL without --download"
    exit 0
fi

case "$RESULTS" in
    /*) ;;
    *) RESULTS="$PWD/$RESULTS" ;;
esac
if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$RESULTS"
fi
if [ -d "$RESULTS" ]; then
    RESULTS="$(cd "$RESULTS" && pwd)"
fi

ENTRY_ARGS=(--model "/models/$MODEL_NAME" --results /results --run-id "$RUN_ID")
if [ "$FULL_WARMUP" -eq 1 ]; then ENTRY_ARGS+=(--full-warmup); fi
if [ -n "$SERVER_TIMEOUT" ]; then ENTRY_ARGS+=(--server-timeout "$SERVER_TIMEOUT"); fi
if [ -n "$VARIANTS" ]; then ENTRY_ARGS+=(--variants "$VARIANTS"); fi
if [ "$WITH_PROFILE" -eq 1 ]; then ENTRY_ARGS+=(--with-profile); fi
if [ -n "$FROM_PHASE" ]; then ENTRY_ARGS+=(--from "$FROM_PHASE"); fi
if [ "$DOWNLOAD" -eq 1 ]; then ENTRY_ARGS+=(--download); fi
if [ "$NO_RECIPE_CACHE" -eq 0 ]; then ENTRY_ARGS+=(--recipe-cache /recipe-cache); fi
if [ "$QUICK_GATES" -eq 1 ]; then ENTRY_ARGS+=(--quick-gates); fi
if [ "$NO_VERIFY_PROBE" -eq 1 ]; then ENTRY_ARGS+=(--no-verify-probe); fi
if [ -n "$LOAD_READ_TIMEOUT" ]; then ENTRY_ARGS+=(--load-read-timeout "$LOAD_READ_TIMEOUT"); fi
if [ -n "$WARM_REQUEST_TIMEOUT" ]; then ENTRY_ARGS+=(--warm-request-timeout "$WARM_REQUEST_TIMEOUT"); fi
# The operator's request only: the orchestrator sets VLLM_HPU_HC_FUSED per server, after the card test.
if [ "$HC_FUSED" -eq 1 ]; then ENTRY_ARGS+=(--hc-fused); fi
# The same for the MoE glue kernels: the request only, VLLM_HPU_MOE_GLUE is the orchestrator's to set.
if [ "$MOE_GLUE" -eq 1 ]; then ENTRY_ARGS+=(--moe-glue); fi
# Appended last so the --dry-run line ends with --no-warm-pass.
if [ "$NO_WARM_PASS" -eq 1 ]; then ENTRY_ARGS+=(--no-warm-pass); fi

# The recipe-cache mount is optional, so it is collected separately and spliced into the command
# with the `${arr[@]+"${arr[@]}"}` guard (bash 3.2 keeps `set -u` quiet for the empty array of
# --no-recipe-cache). PT_HPU_RECIPE_CACHE_CONFIG is not set here: the orchestrator hands it per
# server (RUK-3242), because the Habana bridge rejects the plugin's colon-list GC_KERNEL_PATH
# next to a recipe cache.
RECIPE_CACHE_ARGS=()
if [ "$NO_RECIPE_CACHE" -eq 0 ]; then
    RECIPE_CACHE_ARGS=(
        -v "$RECIPE_CACHE:/recipe-cache"
    )
fi

# --no-combined-kernel-db puts FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0 into the container, where the
# loader treats the combined kernel database as absent (GC_KERNEL_PATH stays a colon list) and the
# orchestrator treats it as not installed (servers with the fused GDN kernel get no recipe cache).
# The install builds the combined database regardless; the variable only stops its use (RUK-3249).
COMBINED_DB_ARGS=()
if [ "$NO_COMBINED_DB" -eq 1 ]; then
    COMBINED_DB_ARGS=(
        -e FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0
    )
fi

# --init: an init process as PID 1 reaps the workers a crashed server leaves behind (without it they
# stay zombies and every crashed server cost the orchestrator its full 65 s stop wait on the
# 2026-11-18 rental). --sig-proxy=false and the run-id label belong to the end path below: the script
# itself signals the container, and it only ever stops a container that carries its own run id.
# PT_HPU_LAZY_ACC_PAR_MODE=0 turns the bridge's parallel graph accumulation off for every process of
# the run (gates, tiny gate, servers): with it on, a custom op (GDN, hyper-connection, MoE glue) whose
# input is produced by a still-queued op gets that input as an empty graph input ("Empty tensor
# optional", "ValidateSyncInputTensors tensor_data is empty"). Card test 2026-11-20: the
# hyper-connection op test is 84 passed, 6 failed with the default 1 and 90 passed with 0.
# PYTHONFAULTHANDLER gives every Python process a stack on a crash; HABANA_LOGS moves the Habana
# runtime logs from the container's home (gone with --rm) into the run folder.
DOCKER_CMD=(
    docker run --rm --name "$NAME"
    --label "fn-rental.run-id=$RUN_ID"
    --init
    --sig-proxy=false
    --runtime=habana
    -e HABANA_VISIBLE_DEVICES=all
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none
    -e PYTHONUNBUFFERED=1
    -e PYTHONFAULTHANDLER=1
    -e PT_HPU_LAZY_ACC_PAR_MODE=0
    -e "HABANA_LOGS=/results/$RUN_ID/habana_logs"
    --cap-add=sys_nice
    --net=host
    --ipc=host
    -v "$SCRIPT_DIR:/bundle"
    -v "$MODEL_PARENT:/models"
    -v "$RESULTS:/results"
    ${RECIPE_CACHE_ARGS[@]+"${RECIPE_CACHE_ARGS[@]}"}
    ${COMBINED_DB_ARGS[@]+"${COMBINED_DB_ARGS[@]}"}
    "$IMAGE"
    bash /bundle/plugin/tools/flash_next/rental/container_entry.sh
    "${ENTRY_ARGS[@]}"
)

if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "${DOCKER_CMD[@]}"
    printf '\n'
    exit 0
fi

# Only a real run populates the cache: a dry run writes nothing.
if [ "$NO_RECIPE_CACHE" -eq 0 ]; then
    mkdir -p "$RECIPE_CACHE"
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not found on this host" >&2
    exit 1
fi

docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"habana"' || {
    echo "error: the habana container runtime is not registered with Docker; run: sudo bash $SCRIPT_DIR/prepare_host.sh" >&2
    exit 1
}

if [ "$NO_DISK_CHECK" -eq 0 ]; then
    DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
    if [ -z "$DOCKER_ROOT" ] || [ ! -d "$DOCKER_ROOT" ]; then
        echo "warning: docker info gave no root folder; skipping its free-space check" >&2
    else
        free="" mnt=""
        read -r free mnt < <(disk_free "$DOCKER_ROOT") || true
        if [ -z "$free" ]; then
            echo "warning: cannot read the free space of $DOCKER_ROOT; continuing" >&2
        elif [ "$free" -lt "$((DOCKER_NEED_GB * 1000000000))" ]; then
            echo "error: Docker's root $DOCKER_ROOT (filesystem mounted at $mnt) has $((free / 1000000000)) GB \
free; the container's installs need $DOCKER_NEED_GB GB. Fix: free space there (docker system prune removes \
stopped containers and unused images), or move Docker's \"data-root\" in /etc/docker/daemon.json to a bigger \
disk and rerun prepare_host.sh, or pass --no-disk-check." >&2
            exit 1
        else
            echo "disk: Docker's root $DOCKER_ROOT has $((free / 1000000000)) GB free: ok"
        fi
    fi
fi

if command -v hl-smi >/dev/null 2>&1; then
    if [ "$(hl-smi -Q index,name -f csv,noheader 2>/dev/null | grep -c . || true)" -eq 0 ]; then
        echo "warning: hl-smi lists no cards; the run will likely fail (check the driver, or reboot and rerun prepare_host.sh)" >&2
    fi
fi

if has_name "$(docker ps --format '{{.Names}}' 2>/dev/null || true)" "$DOWNLOAD_NAME"; then
    echo "error: the checkpoint download (container $DOWNLOAD_NAME) is still running; wait for it: sudo docker wait $DOWNLOAD_NAME (prints 0 when the checkpoint is complete; progress: sudo docker logs --tail 3 $DOWNLOAD_NAME)" >&2
    exit 1
fi

# Everything the run prints is also kept in the results folder (so in the tarball), to be read
# or pasted after the fact; tee -a keeps one file across --from resumes of the same run id.
RUN_DIR="$RESULTS/$RUN_ID"
CONSOLE_LOG="$RUN_DIR/console.log"
TARBALL="$RESULTS/$RUN_ID.tar.gz"
mkdir -p "$RUN_DIR/habana_logs"  # HABANA_LOGS of the container

# --- the end path -------------------------------------------------------------------------------
# One function ends the script, reached from the normal end and from a signal: it stops the
# container if it is still there, collects the evidence, writes the tarball and prints the two
# closing lines. Everything in it is best effort and bounded, so the tarball is always written.

# Seconds the orchestrator gets to stop its servers and write the report before docker stop.
STOP_WAIT_S="${FN_RENTAL_STOP_WAIT_S:-90}"
RUN_PID=""
FINISHING=0

# A line for the operator and for console.log (after an SSH drop only the file is left).
note() {
    echo "$@" | tee -a "$CONSOLE_LOG"
}

# True while this run's container exists: fn-rental carrying this run id as its label, so a
# container that another run started under the same name is never touched.
container_exists() {
    [ "$(docker inspect --format '{{index .Config.Labels "fn-rental.run-id"}}' "$NAME" 2>/dev/null)" = "$RUN_ID" ]
}

# wait_gone SECONDS — 0 as soon as the container is gone, 1 when it is still there after SECONDS.
wait_gone() {
    local left="$1"
    while container_exists; do
        if [ "$left" -le 0 ]; then
            return 1
        fi
        sleep 1
        left=$((left - 1))
    done
    return 0
}

# Stop the container in bounded steps: SIGINT for the orchestrator, which then stops its servers
# and writes the report (STOP_WAIT_S seconds for that); docker stop (SIGTERM, SIGKILL after 10 s);
# docker kill. During the install steps there is no orchestrator, and docker stop comes at once.
stop_container() {
    if ! container_exists; then
        return 0
    fi
    if docker exec "$NAME" pkill -INT -f tools.flash_next.rental.main >/dev/null 2>&1; then
        note "stopping container $NAME: the orchestrator was asked to stop its servers and write the report (at most $STOP_WAIT_S s)"
        if wait_gone "$STOP_WAIT_S"; then
            return 0
        fi
        note "container $NAME is still running after $STOP_WAIT_S s: docker stop"
    else
        note "stopping container $NAME (no orchestrator is running in it): docker stop"
    fi
    docker stop -t 10 "$NAME" >/dev/null 2>&1
    if wait_gone 5; then
        return 0
    fi
    note "container $NAME did not stop: docker kill"
    docker kill "$NAME" >/dev/null 2>&1
    if ! wait_gone 5; then
        note "WARNING: container $NAME is still there and may hold the cards; remove it: sudo docker rm -f $NAME"
    fi
}

# The docker client and tee end when the container is gone. Give them 15 s to copy its last lines
# into console.log, then end what is left of them, so a stuck client cannot hold the tarball back.
wait_client() {
    local left=15
    if [ -z "$RUN_PID" ]; then
        return 0
    fi
    while kill -0 "$RUN_PID" 2>/dev/null; do
        if [ "$left" -le 0 ]; then
            pkill -P "$RUN_PID" 2>/dev/null
            kill "$RUN_PID" 2>/dev/null
            break
        fi
        sleep 1
        left=$((left - 1))
    done
    wait "$RUN_PID" 2>/dev/null
}

# The state of the container (ExitCode, OOMKilled, StartedAt) when it is still there. --rm removes a
# finished container, so this records one that outlived its docker client or was interrupted.
capture_container_state() {
    local state=""
    if container_exists; then
        state="$(docker inspect --format '{{json .State}}' "$NAME" 2>/dev/null)" || state=""
        if [ -n "$state" ]; then
            printf '%s\n' "$state" >"$RUN_DIR/container-state.json"
        fi
    fi
}

# Evidence that is gone once the node is: the kernel log tail (OOM kills, driver errors; it needs
# root on most hosts and is skipped quietly without it) and the digest of the image the run used.
capture_evidence() {
    local text="" digest="" image_id=""
    text="$(dmesg 2>/dev/null | tail -n 400)" || text=""
    if [ -n "$text" ]; then
        printf '%s\n' "$text" >"$RUN_DIR/dmesg-tail.txt"
    fi
    digest="$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null)" || digest=""
    image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null)" || image_id=""
    if [ -n "$digest$image_id" ]; then
        {
            echo "image: $IMAGE"
            echo "repo digest: ${digest:-none}"
            echo "image id: ${image_id:-unknown}"
        } >"$RUN_DIR/image-digest.txt"
    fi
}

# finish EXIT_CODE — the end of every run. Signals are ignored from here on, so it runs once and to
# its end.
finish() {
    local rc="$1" tar_rc=0
    FINISHING=1
    trap '' INT TERM HUP PIPE
    set +e
    capture_container_state
    stop_container
    wait_client
    capture_evidence
    # The tiny gate's weights (tiny/ckpt/*.safetensors, about 136 MB, rebuilt by every run) stay out
    # of the tarball; the checkpoint's config and reference.json go in.
    tar czf "$TARBALL" --exclude='tiny/ckpt/*.safetensors' -C "$RESULTS" "$RUN_ID"
    tar_rc=$?
    if [ "$tar_rc" -ne 0 ]; then
        echo "WARNING: tar ended with exit code $tar_rc; if $TARBALL is missing or cut short, send the folder $RUN_DIR" >&2
        if [ "$rc" -eq 0 ]; then
            rc="$tar_rc"
        fi
    fi
    echo "PASTE THIS FILE: $RESULTS/$RUN_ID/summary.md (plus the last 40 lines of $CONSOLE_LOG if the run stopped early)"
    echo "SEND THIS FILE: $TARBALL"
    exit "$rc"
}

# on_signal NAME EXIT_CODE — Ctrl-C, SIGTERM, or SIGHUP from a dropped SSH session.
on_signal() {
    if [ "$FINISHING" -eq 1 ]; then
        return 0
    fi
    FINISHING=1
    trap '' INT TERM HUP PIPE
    set +e
    note "interrupted by SIG$1: stopping the run's container, then writing the tarball (signals are ignored from here on)"
    finish "$2"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

echo "run id: $RUN_ID" | tee -a "$CONSOLE_LOG"
set +e
# The container runs in the background and the script waits for it with `wait`, which a signal
# interrupts at once (a foreground command would hold the trap back until the container ends).
(
    # The docker client and tee ignore Ctrl-C and a dropped terminal: on_signal stops the container,
    # and they go on copying its last lines into console.log until it is gone.
    trap '' INT HUP
    "${DOCKER_CMD[@]}" 2>&1 | tee -a "$CONSOLE_LOG"
    rc="${PIPESTATUS[0]}"
    # --init needs Docker's docker-init binary on the host. A host without it refuses the container
    # within a second (exit 125, the daemon's message names docker-init): start it once more without
    # the flag instead of losing the run to it (a crashed server then takes about 65 s longer to stop).
    if [ "$rc" -eq 125 ] && tail -n 5 "$CONSOLE_LOG" | grep -q "docker-init"; then
        echo "warning: this host's Docker has no docker-init; starting the container again without --init" \
            | tee -a "$CONSOLE_LOG"
        NO_INIT_CMD=()
        for arg in "${DOCKER_CMD[@]}"; do
            [ "$arg" = "--init" ] || NO_INIT_CMD+=("$arg")
        done
        "${NO_INIT_CMD[@]}" 2>&1 | tee -a "$CONSOLE_LOG"
        rc="${PIPESTATUS[0]}"
    fi
    exit "$rc"
) &
RUN_PID=$!
wait "$RUN_PID"
RC=$?
echo "container exit code: $RC" | tee -a "$CONSOLE_LOG"
finish "$RC"
