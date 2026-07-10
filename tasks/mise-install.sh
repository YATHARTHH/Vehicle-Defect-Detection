#!/bin/bash

is_windows_bash() {
  case "$(uname -s)" in
    MINGW*|CYGWIN*|MSYS*) return 0 ;;
    *) return 1 ;;
  esac
}

install_mise_windows() {
  echo "Detected Windows (Git Bash). Installing mise via winget..."
  powershell.exe -NoProfile -Command "winget install --id jdx.mise --exact --source winget --accept-source-agreements --accept-package-agreements"

  # Find the WinGet portable executable or link and expose it in this session
  local mise_win_bin
  mise_win_bin=$(powershell.exe -NoProfile -Command \
    "\$cmd = Get-Command mise -ErrorAction SilentlyContinue; if (\$cmd) { Split-Path -Parent \$cmd.Source; exit }; \$link = Join-Path \$env:LOCALAPPDATA 'Microsoft\WinGet\Links\mise.exe'; if (Test-Path -LiteralPath \$link) { Split-Path -Parent \$link; exit }; \$packages = Join-Path \$env:LOCALAPPDATA 'Microsoft\WinGet\Packages'; if (Test-Path -LiteralPath \$packages) { \$mise = Get-ChildItem -LiteralPath \$packages -Recurse -Filter 'mise.exe' -ErrorAction SilentlyContinue | Select-Object -First 1; if (\$mise) { Split-Path -Parent \$mise.FullName } }" \
    2>/dev/null | tr -d '\r')

  if [ -n "$mise_win_bin" ]; then
    local mise_unix_bin
    mise_unix_bin="$(cygpath -u "$mise_win_bin")"
    export PATH="$mise_unix_bin:$PATH"

    # Persist to ~/.bashrc so future Git Bash sessions have mise in PATH
    local bashrc_entry="export PATH=\"$mise_unix_bin:\$PATH\""
    grep -qxF "$bashrc_entry" ~/.bashrc 2>/dev/null || echo "$bashrc_entry" >> ~/.bashrc
    echo "Added mise to PATH. Run: export PATH=\"$mise_unix_bin:\$PATH\" to use it in this session."
  else
    echo "ERROR: Could not locate mise after WinGet install. Open a new terminal and re-run."
    return 1
  fi
}

install_mise_unix() {
  # The mise installer detects $SHELL and updates the correct config (~/.zshrc, ~/.bashrc, etc.)
  curl https://mise.run | sh
  # Set PATH for this session so mise setup can run immediately
  export PATH="$HOME/.local/bin:$PATH"
}

if command -v mise &>/dev/null; then
  mise trust
  mise --version
else
  if is_windows_bash; then
    install_mise_windows
  else
    install_mise_unix
  fi

  mise trust
  mise --version
fi

mise setup

# Ensure the hooks are executable - Git ignores non-executable hooks on macOS/Linux.
chmod +x .githooks/* 2>/dev/null || true
