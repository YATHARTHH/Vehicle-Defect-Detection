$ErrorActionPreference = "Stop"

function Add-ToPathForSession {
    param([string]$PathToAdd)

    if (-not $PathToAdd -or -not (Test-Path -LiteralPath $PathToAdd)) {
        return
    }

    $pathParts = $env:PATH -split ';'
    if ($pathParts -notcontains $PathToAdd) {
        $env:PATH = "$PathToAdd;$env:PATH"
    }
}

function Add-ToUserPath {
    param([string]$PathToAdd)

    if (-not $PathToAdd -or -not (Test-Path -LiteralPath $PathToAdd)) {
        return
    }

    $userPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
    $pathParts = @($userPath -split ';' | Where-Object { $_ })
    if ($pathParts -notcontains $PathToAdd) {
        $newUserPath = if ($userPath) { "$PathToAdd;$userPath" } else { $PathToAdd }
        [System.Environment]::SetEnvironmentVariable("PATH", $newUserPath, "User")
        Write-Host "Added mise to user PATH: $PathToAdd"
    }
}

function Find-MiseExecutable {
    $command = Get-Command mise -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $wingetLinksMise = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\mise.exe"
    if (Test-Path -LiteralPath $wingetLinksMise) {
        return $wingetLinksMise
    }

    $wingetPackages = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path -LiteralPath $wingetPackages) {
        $packageMise = Get-ChildItem -LiteralPath $wingetPackages -Recurse -Filter "mise.exe" -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($packageMise) {
            return $packageMise.FullName
        }
    }

    return $null
}

function Use-MiseExecutable {
    $miseExe = Find-MiseExecutable
    if (-not $miseExe) {
        throw "mise was installed, but mise.exe could not be found. Open a new PowerShell session and re-run .\tasks\mise-install.cmd."
    }

    $miseDir = Split-Path -Parent $miseExe
    Add-ToPathForSession $miseDir
    Add-ToUserPath $miseDir
    return $miseExe
}

if (Get-Command mise -ErrorAction SilentlyContinue) {
    mise trust
    mise --version
}
else {
    $originalPath = Get-Location

    winget install --id jdx.mise --exact --source winget --accept-source-agreements --accept-package-agreements

    Use-MiseExecutable | Out-Null

    Set-Location $originalPath

    mise trust
    mise --version
}

mise setup

# Add Python Scripts dir to PATH so pre-commit binary is accessible in this session and future ones
$pythonScripts = mise exec -- python -c "import sys, os; print(os.path.join(os.path.dirname(sys.executable), 'Scripts'))" 2>$null
if ($pythonScripts -and (Test-Path $pythonScripts)) {
    $env:PATH = "$pythonScripts;$env:PATH"
    $userPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if ($userPath -notlike "*$pythonScripts*") {
        [System.Environment]::SetEnvironmentVariable("PATH", "$pythonScripts;$userPath", "User")
        Write-Host "Added Python Scripts to PATH: $pythonScripts"
    }
}
