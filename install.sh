#!/bin/bash
# One-command installer: Inkling 2 Mini on an Apple Silicon Mac (tuned for a 16 GB M4 Mac mini),
# with Ternary Bonsai 2 27B as an optional second model to compare against.
#
#   curl -fsSL https://raw.githubusercontent.com/TechGuardCoders/inkling2-mini-mac-recipe/main/install.sh | bash
#   ... | bash -s -- --models inkling,bonsai     # both models (one runs at a time)
#
# Options:
#   --models LIST       inkling (default, MLX), inkling-gguf (llama.cpp), bonsai; comma-separated
#   --prefix DIR        install here (default ~/.mini-llm)
#   --port N            server port (default 8080, loopback only)
#   --no-start          install only, don't start the server
#   --no-wired-limit    don't offer to raise the GPU memory limit
#   --no-path           don't add mini-llm to your PATH
#   --dry-run           check the Mac and print the plan, change nothing
set -euo pipefail

REPO="TechGuardCoders/inkling2-mini-mac-recipe"
BRANCH="${MINI_LLM_BRANCH:-main}"
PREFIX="${MINI_LLM_PREFIX:-$HOME/.mini-llm}"
MODELS="inkling"; PORT=""; START=1; WIRED=1; ADDPATH=1; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS="$2"; shift ;;
    --prefix) PREFIX="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --no-start) START=0 ;;
    --no-wired-limit) WIRED=0 ;;
    --no-path) ADDPATH=0 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
MODELS="$(echo "$MODELS" | tr ',' ' ')"

# This recipe grew into Nuemac (github.com/TechGuardCoders/nuemac), which serves Inkling 2 Mini on its own
# engine. The default install forwards there; --models inkling-gguf / bonsai still install the comparison
# builds below.
if [ "$MODELS" = "inkling" ] && [ "${MINI_LLM_LEGACY:-0}" != 1 ]; then
  FWD=()
  [ -n "$PORT" ] && FWD+=(--port "$PORT")
  [ "$START" = 0 ] && FWD+=(--no-start)
  [ "$WIRED" = 0 ] && FWD+=(--no-wired-limit)
  [ "$ADDPATH" = 0 ] && FWD+=(--no-path)
  [ "$DRY" = 1 ] && FWD+=(--dry-run)
  printf '[1m%s[0m
' "This recipe is now Nuemac: installing it (github.com/TechGuardCoders/nuemac)"
  NUEMAC_INSTALL="$(mktemp)"
  curl -fsSL https://raw.githubusercontent.com/TechGuardCoders/nuemac/main/install.sh -o "$NUEMAC_INSTALL"     || { echo "could not download the Nuemac installer" >&2; exit 1; }
  exec bash "$NUEMAC_INSTALL" "${FWD[@]+"${FWD[@]}"}"
fi

b() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
ask() { # question -> 0 for yes; needs a terminal (works under curl | bash via /dev/tty)
  local a=""; [ -r /dev/tty ] || return 1
  printf '%s [y/N] ' "$1" > /dev/tty; read -r a < /dev/tty || return 1
  case "$a" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}
gib() { awk -v b="$1" 'BEGIN{printf "%.1f GB", b/1e9}'; }

# ---- 1. check the Mac -------------------------------------------------------------------------
b "mini-llm installer"
[ "$(uname -s)" = Darwin ] || die "this installer is for macOS"
[ "$(uname -m)" = arm64 ] || die "needs Apple Silicon (M1 or newer)"
OSMAJ="$(sw_vers -productVersion | cut -d. -f1)"
[ "$OSMAJ" -ge 14 ] || die "needs macOS 14 or newer (you have $(sw_vers -productVersion))"
MEM_MB=$(( $(sysctl -n hw.memsize) / 1048576 ))
[ "$MEM_MB" -ge 15000 ] || die "needs 16 GB of memory or more"
CHIP="$(sysctl -n machdep.cpu.brand_string)"
echo "Mac        $CHIP, $((MEM_MB / 1024)) GB, macOS $(sw_vers -productVersion)"

# ---- 2. get the recipe files --------------------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -n "$HERE" ] && [ -f "$HERE/engines.conf" ] && [ -d "$HERE/models" ]; then
  RECIPE_SRC="$HERE"
else
  RECIPE_SRC="$(mktemp -d)/recipe"; mkdir -p "$RECIPE_SRC"
  curl -fsSL "https://codeload.github.com/$REPO/tar.gz/refs/heads/$BRANCH" | tar xz -C "$RECIPE_SRC" --strip-components 1 \
    || die "could not download the recipe from github.com/$REPO"
fi

present() { # conf -> 0 if that model's files are already installed and verified
  ( . "$1"
    if [ "${ENGINE_TYPE:-llama}" = mlx ]; then [ -f "$PREFIX/models/$MODEL_DIR/.verified" ]
    else [ -f "$PREFIX/models/$MODEL_FILE" ]; fi )
}
BREW="$(command -v brew || { [ -x /opt/homebrew/bin/brew ] && echo /opt/homebrew/bin/brew; } || true)"
# The MLX build needs oMLX's Python stack from Homebrew; without Homebrew, use the llama.cpp build.
if [ -z "$BREW" ] && echo " $MODELS " | grep -q ' inkling '; then
  echo "Homebrew not found: installing the llama.cpp build of Inkling (inkling-gguf) instead."
  echo "  (For the faster MLX build, install Homebrew from https://brew.sh and run this again.)"
  MODELS="$(echo " $MODELS " | sed 's/ inkling / inkling-gguf /' | xargs)"
fi

NEED=0; PLAN=""
for m in $MODELS; do
  [ -f "$RECIPE_SRC/models/$m.conf" ] || die "unknown model '$m' (choose: $(ls "$RECIPE_SRC/models" | sed 's/.conf//' | tr '\n' ' '))"
  present "$RECIPE_SRC/models/$m.conf" || NEED=$(( NEED + $(. "$RECIPE_SRC/models/$m.conf"
    have=0  # count what an interrupted earlier run already downloaded
    if [ "${ENGINE_TYPE:-llama}" = mlx ] && [ -d "$PREFIX/models/$MODEL_DIR" ]; then
      have=$(( $(du -sk "$PREFIX/models/$MODEL_DIR" | awk '{print $1}') * 1024 ))
    elif [ -f "$PREFIX/models/$MODEL_FILE.part" ]; then
      have=$(stat -f %z "$PREFIX/models/$MODEL_FILE.part")
    fi
    left=$(( MODEL_BYTES - have )); [ "$left" -gt 0 ] || left=0; echo "$left") / 1000000 ))
  PLAN="$PLAN $(. "$RECIPE_SRC/models/$m.conf"; echo "$MODEL_LABEL [$(gib "$MODEL_BYTES")]"),"
done
omlx_python() { for p in /opt/homebrew/opt/omlx/libexec/bin/python /usr/local/opt/omlx/libexec/bin/python; do
  [ -x "$p" ] && { echo "$p"; return 0; }; done; return 1; }
if echo " $MODELS " | grep -q ' inkling ' && ! omlx_python >/dev/null; then
  NEED=$(( NEED + 2000 )); PLAN="$PLAN oMLX 0.8.2 from Homebrew [~2 GB],"   # Python stack for inkserve
fi
mkdir -p "$PREFIX" 2>/dev/null || true
FREE_MB=$(( $(df -k "$PREFIX" | awk 'NR==2{print $4}') / 1024 ))
echo "Install to $PREFIX  (free disk $((FREE_MB / 1024)) GB, download ~$((NEED / 1000)) GB)"
echo "Models    ${PLAN%,}"
[ "$FREE_MB" -gt $(( NEED + 3000 )) ] || die "not enough free disk space"
if [ "$DRY" = 1 ]; then echo "dry run: nothing changed"; exit 0; fi

mkdir -p "$PREFIX"/{bin,engines,models,logs,results}
if [ "$(cd "$RECIPE_SRC" && pwd -P)" != "$(cd "$PREFIX" && pwd -P)/recipe" ]; then
  rm -rf "$PREFIX/recipe.new"; cp -R "$RECIPE_SRC" "$PREFIX/recipe.new"
  rm -rf "$PREFIX/recipe"; mv "$PREFIX/recipe.new" "$PREFIX/recipe"
fi
install -m 755 "$PREFIX/recipe/bin/mini-llm" "$PREFIX/bin/mini-llm"
[ -n "$PORT" ] && echo "PORT=$PORT" > "$PREFIX/settings"
[ -f "$PREFIX/active" ] || echo "$(echo "$MODELS" | awk '{print $1}')" > "$PREFIX/active"

verify() { # file sha256
  printf "  verifying %s ... " "$(basename "$1")"
  if [ "$(shasum -a 256 "$1" | awk '{print $1}')" = "$2" ]; then echo ok; return 0; fi
  echo "checksum mismatch"; return 1
}
fetch() { # url dest sha256
  local part="$2.part"
  if [ -f "$2" ]; then verify "$2" "$3" && return 0; mv "$2" "$part"; fi
  echo "  downloading $(basename "$2")"
  for _ in 1 2 3 4 5; do
    curl -fL --retry 5 --retry-delay 5 -C - --progress-bar -o "$part" "$1" && break
    echo "  download interrupted, resuming..."; sleep 5
  done
  verify "$part" "$3" || { rm -f "$part"; die "download of $(basename "$2") was corrupt; run the installer again"; }
  mv "$part" "$2"
}

# ---- 3. engines -----------------------------------------------------------------------------------
b "Engines"
for m in $MODELS; do (. "$PREFIX/recipe/models/$m.conf"; echo "${ENGINE_TYPE:-llama} ${OMLX_VERSION:-}"); done |
  { grep '^mlx' || true; } | sort -u | while read -r _ want; do
  # oMLX (Apache-2.0) supplies MLX, mlx-lm and the Inkling 2 model code that engine/inkserve.py runs on.
  if ! omlx_python >/dev/null; then
    echo "  installing oMLX from Homebrew (brew install jundot/omlx/omlx)"
    "$BREW" install jundot/omlx/omlx
  fi
  py="$(omlx_python)" || die "oMLX installed but its Python was not found"
  have="$("$py" -I -c 'import importlib.metadata as m; print(m.version("omlx"))' 2>/dev/null || echo unknown)"
  if [ "$have" = "$want" ]; then echo "  oMLX $have (MLX engine for inkserve)"
  else echo "  oMLX $have installed; this recipe was tested with $want (it checks itself at start-up)"; fi
done
for m in $MODELS; do (. "$PREFIX/recipe/models/$m.conf"; [ "${ENGINE_TYPE:-llama}" = llama ] && echo "$ENGINE"; true); done |
  sort -u | while read -r e; do
  line="$(grep "^$e|" "$PREFIX/recipe/engines.conf")" || die "engine $e not in engines.conf"
  IFS='|' read -r name url sha dir <<< "$line"
  if [ -x "$PREFIX/engines/$name/llama-server" ]; then echo "  $name already installed"; continue; fi
  fetch "$url" "$PREFIX/engines/$name.tar.gz" "$sha"
  rm -rf "$PREFIX/engines/$name.tmp"; mkdir -p "$PREFIX/engines/$name.tmp"
  tar xzf "$PREFIX/engines/$name.tar.gz" -C "$PREFIX/engines/$name.tmp"
  rm -rf "$PREFIX/engines/$name"; mv "$PREFIX/engines/$name.tmp/$dir" "$PREFIX/engines/$name"
  rm -rf "$PREFIX/engines/$name.tmp" "$PREFIX/engines/$name.tar.gz"
  xattr -dr com.apple.quarantine "$PREFIX/engines/$name" 2>/dev/null || true
  "$PREFIX/engines/$name/llama-server" --version 2>&1 | grep -m1 version | sed 's/^/  /'
done

# ---- 4. models -----------------------------------------------------------------------------------
b "Models"
for m in $MODELS; do
  ( . "$PREFIX/recipe/models/$m.conf"
    if [ "${ENGINE_TYPE:-llama}" = mlx ]; then
      present "$PREFIX/recipe/models/$m.conf" && { echo "  $MODEL_DIR already installed"; exit 0; }
      mkdir -p "$PREFIX/models/$MODEL_DIR"
      for f in $MODEL_FILES; do
        fetch "https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REV/${f%%:*}" "$PREFIX/models/$MODEL_DIR/${f%%:*}" "${f##*:}"
      done
      echo "$MODEL_REPO@$MODEL_REV" > "$PREFIX/models/$MODEL_DIR/.verified"
    else
      fetch "$MODEL_URL" "$PREFIX/models/$MODEL_FILE" "$MODEL_SHA256"
    fi )
done

# ---- 5. GPU memory limit (16 GB Macs) ---------------------------------------------------------
# macOS lets the GPU wire only ~2/3 of memory by default (~10.7 GB on 16 GB). Inkling needs ~14 GB.
LIM="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)"
if [ "$WIRED" = 1 ] && [ "$MEM_MB" -le 17000 ] && [ "$LIM" -lt 14500 ]; then
  b "GPU memory limit"
  echo "  On a 16 GB Mac the GPU may use only ~10.7 GB by default; this setup needs 14.5 GB."
  echo "  Raising it needs your password (sudo). Close other big apps while the model runs."
  if ask "  Raise it to 14500 MB now?"; then
    sudo sysctl iogpu.wired_limit_mb=14500 < /dev/tty
    if ask "  Keep it after reboot (installs a small boot job in /Library/LaunchDaemons)?"; then
      sudo tee /Library/LaunchDaemons/com.mini-llm.wiredlimit.plist > /dev/null < "$PREFIX/recipe/support/com.mini-llm.wiredlimit.plist"
      sudo launchctl bootstrap system /Library/LaunchDaemons/com.mini-llm.wiredlimit.plist 2>/dev/null || true
    fi
  else
    echo "  Skipped. Before starting, run:  sudo sysctl iogpu.wired_limit_mb=14500"
  fi
fi

# ---- 6. PATH ------------------------------------------------------------------------------------
if [ "$ADDPATH" = 1 ]; then
  mkdir -p "$HOME/.local/bin"; ln -sf "$PREFIX/bin/mini-llm" "$HOME/.local/bin/mini-llm"
  if ! grep -qs 'mini-llm' "$HOME/.zprofile"; then
    printf '\n# mini-llm\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$HOME/.zprofile"
  fi
fi

# ---- 7. start -------------------------------------------------------------------------------------
if [ "$START" = 1 ]; then
  b "Starting $(cat "$PREFIX/active")"
  MINI_LLM_PREFIX="$PREFIX" "$PREFIX/bin/mini-llm" start
  MINI_LLM_PREFIX="$PREFIX" "$PREFIX/bin/mini-llm" bench --qa-only | grep "quality check"
fi

b "Done"
cat <<EOF
  mini-llm status            what's installed and running
  mini-llm bench             measure speed + quality on your Mac (~10 min)
  mini-llm use bonsai        switch to Ternary Bonsai 2 27B (downloads it the first time)
  mini-llm use inkling-gguf  the llama.cpp build of Inkling, to compare engines
  mini-llm stop | start      the server also starts by itself at login
  API (OpenAI-compatible):   http://127.0.0.1:${PORT:-8080}/v1
  Open a new terminal (or run: export PATH="\$HOME/.local/bin:\$PATH") to use mini-llm.
EOF
