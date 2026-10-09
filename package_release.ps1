param(
    [string]$Version,
    [string]$WslDistro = 'Ubuntu-22.04'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $Version) {
    $VersionHeader = Get-Content -Raw (Join-Path $Root 'version.h')
    if ($VersionHeader -notmatch 'TARI_MINER_VERSION\s+"([^"]+)"') {
        throw 'Could not read the miner version from version.h.'
    }
    $Version = $Matches[1]
}

$Dist = Join-Path $Root 'dist'
$StageRoot = Join-Path $Dist "staging\v$Version"
$ReleaseRoot = Join-Path $Dist 'release'
$RootFull = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
$StageFull = [IO.Path]::GetFullPath($StageRoot)
$ReleaseFull = [IO.Path]::GetFullPath($ReleaseRoot)
if (-not $StageFull.StartsWith($RootFull, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to stage outside the repository: $StageFull"
}
if (-not $ReleaseFull.StartsWith($RootFull, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to package outside the repository: $ReleaseFull"
}

if (Test-Path -LiteralPath $StageRoot) {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force
}
if (Test-Path -LiteralPath $ReleaseRoot) {
    Remove-Item -LiteralPath $ReleaseRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $StageRoot, $ReleaseRoot | Out-Null

$Packages = @(
    @{
        Name = "TARI.Miner-v$Version-windows"
        Readme = 'packaging\README.txt'
        Starter = @('start-c29.bat', 'start-c29.ps1')
        Suffix = '.exe'
    },
    @{
        Name = "TARI.Miner-v$Version-linux"
        Readme = 'packaging\README-LINUX.txt'
        Starter = @('start-c29.sh')
        Suffix = ''
    }
)
$Architectures = @('sm_86', 'sm_89', 'sm_120')

foreach ($Package in $Packages) {
    $Stage = Join-Path $StageRoot $Package.Name
    $StageBin = Join-Path $Stage 'bin'
    New-Item -ItemType Directory -Force -Path $StageBin | Out-Null
    Copy-Item -LiteralPath (Join-Path $Root $Package.Readme) -Destination (Join-Path $Stage 'README.txt')
    foreach ($StarterFile in $Package.Starter) {
        Copy-Item -LiteralPath (Join-Path $Root $StarterFile) -Destination $Stage
    }
    Copy-Item -LiteralPath (Join-Path $Root 'LICENSE') -Destination $Stage
    Copy-Item -LiteralPath (Join-Path $Root 'THIRD_PARTY_NOTICES.md') -Destination $Stage

    foreach ($Arch in $Architectures) {
        foreach ($Program in @('tari_c29_pool_miner', 'tari_c29_solver')) {
            $Binary = Join-Path $Root "bin\${Program}_${Arch}$($Package.Suffix)"
            if (-not (Test-Path -LiteralPath $Binary)) {
                throw "Missing release binary: $Binary"
            }
            Copy-Item -LiteralPath $Binary -Destination $StageBin
        }
    }
}

$HiveName = 'tari-miner-hiveos'
$HiveStage = Join-Path $StageRoot $HiveName
$HiveBin = Join-Path $HiveStage 'bin'
New-Item -ItemType Directory -Force -Path $HiveBin | Out-Null
Copy-Item -LiteralPath (Join-Path $Root 'packaging\README-HIVEOS.txt') -Destination (Join-Path $HiveStage 'README.txt')
Copy-Item -LiteralPath (Join-Path $Root 'start-c29.sh') -Destination $HiveStage
Copy-Item -LiteralPath (Join-Path $Root 'LICENSE') -Destination $HiveStage
Copy-Item -LiteralPath (Join-Path $Root 'THIRD_PARTY_NOTICES.md') -Destination $HiveStage
foreach ($File in @('h-manifest.conf', 'h-config.sh', 'h-run.sh', 'h-stats.sh')) {
    Copy-Item -LiteralPath (Join-Path $Root "hiveos\$File") -Destination $HiveStage
}
foreach ($Arch in $Architectures) {
    foreach ($Program in @('tari_c29_pool_miner', 'tari_c29_solver')) {
        $Binary = Join-Path $Root "bin\${Program}_${Arch}"
        if (-not (Test-Path -LiteralPath $Binary)) {
            throw "Missing HiveOS release binary: $Binary"
        }
        Copy-Item -LiteralPath $Binary -Destination $HiveBin
    }
}

function ConvertTo-WslPath([string]$Path) {
    $FullPath = [IO.Path]::GetFullPath($Path)
    $RootPath = [IO.Path]::GetPathRoot($FullPath)
    if ($RootPath -notmatch '^([A-Za-z]):\\$') {
        throw "Only drive-letter paths can be packaged through WSL: $FullPath"
    }
    $Drive = $Matches[1].ToLowerInvariant()
    $Relative = $FullPath.Substring($RootPath.Length).Replace('\', '/')
    return "/mnt/$Drive/$Relative"
}

function New-WslTarArchive(
    [string]$SourceDirectory,
    [string]$ArchivePath,
    [string]$TopLevel,
    [string[]]$ExecutablePaths
) {
    $SourceWsl = ConvertTo-WslPath $SourceDirectory
    $ArchiveWsl = ConvertTo-WslPath $ArchivePath
    $HelperPath = Join-Path $StageRoot 'create-tar.sh'
    $HelperWsl = ConvertTo-WslPath $HelperPath
    $BashScript = @'
set -euo pipefail
source_dir="$1"
archive="$2"
root_name="$3"
shift 3
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
if [[ "$root_name" == "." ]]; then
    destination="$tmp/package"
else
    destination="$tmp/$root_name"
fi
mkdir -p "$destination"
cp -a "$source_dir/." "$destination/"
find "$destination" -type d -exec chmod 755 {} +
find "$destination" -type f -exec chmod 644 {} +
for relative_path in "$@"; do
    chmod 755 "$destination/$relative_path"
done
if [[ "$root_name" == "." ]]; then
    tar -czf "$archive" -C "$destination" .
else
    tar -czf "$archive" -C "$tmp" "$root_name"
fi
'@
    ($BashScript -replace "`r`n", "`n") | Set-Content -LiteralPath $HelperPath -Encoding ascii -NoNewline
    try {
        $WslArgs = @('-d', $WslDistro, '--', 'bash', $HelperWsl, $SourceWsl, $ArchiveWsl, $TopLevel) + $ExecutablePaths
        $Output = & wsl.exe @WslArgs 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "WSL tar failed for $ArchivePath`n$Output"
        }
    }
    finally {
        Remove-Item -LiteralPath $HelperPath -Force -ErrorAction SilentlyContinue
    }
}

$WindowsArchive = Join-Path $ReleaseRoot "TARI.Miner-v$Version-windows.zip"
$LinuxArchive = Join-Path $ReleaseRoot "TARI.Miner-v$Version-linux.tar.gz"
$HiveArchive = Join-Path $ReleaseRoot "$HiveName-$Version.tar.gz"
Remove-Item -LiteralPath $WindowsArchive, $LinuxArchive, $HiveArchive -Force -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $StageRoot "TARI.Miner-v$Version-windows\*") -DestinationPath $WindowsArchive -CompressionLevel Optimal
New-WslTarArchive `
    -SourceDirectory (Join-Path $StageRoot "TARI.Miner-v$Version-linux") `
    -ArchivePath $LinuxArchive `
    -TopLevel '.' `
    -ExecutablePaths (@('start-c29.sh') + ($Architectures | ForEach-Object { "bin/tari_c29_pool_miner_$_"; "bin/tari_c29_solver_$_" }))
New-WslTarArchive `
    -SourceDirectory $HiveStage `
    -ArchivePath $HiveArchive `
    -TopLevel $HiveName `
    -ExecutablePaths (@('start-c29.sh', 'h-config.sh', 'h-run.sh', 'h-stats.sh') + ($Architectures | ForEach-Object { "bin/tari_c29_pool_miner_$_"; "bin/tari_c29_solver_$_" }))

$Archives = @($WindowsArchive, $LinuxArchive, $HiveArchive)
$Checksums = foreach ($Archive in $Archives) {
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
    "$Hash  $(Split-Path -Leaf $Archive)"
}
$ChecksumsPath = Join-Path $ReleaseRoot 'SHA256SUMS.txt'
$ChecksumText = ($Checksums -join "`n") + "`n"
[IO.File]::WriteAllText($ChecksumsPath, $ChecksumText, [Text.Encoding]::ASCII)

$ReleaseNotes = @'
# TARI.Miner C29 v{0}

Download one package for your operating system. All packages automatically
detect every supported NVIDIA GPU and choose the matching RTX 30, RTX 40, or
RTX 50 backend for each card, including mixed-card rigs.

## Windows

1. Download and extract `TARI.Miner-v{0}-windows.zip`.
2. Run `start-c29.bat`.
3. Enter your Tari wallet address.

## Linux

1. Download and extract `TARI.Miner-v{0}-linux.tar.gz`.
2. Run `./start-c29.sh` and enter your Tari wallet address.

## HiveOS

Use a Custom miner with name `tari-miner-hiveos` and installation URL:

`https://github.com/tari-project/TARI.Miner/releases/download/v{0}/tari-miner-hiveos-{0}.tar.gz`

Set the wallet template to `%WAL%.%WORKER_NAME%`, pool URL to
`stratum+tcp://taric29-ca.luckypool.io:3111`, and pass to `x`.

The pool is prefilled as `taric29-ca.luckypool.io:3111`. There is no developer
fee, and the launcher does not change GPU power, clock, voltage, or fan settings.
'@ -f $Version
$ReleaseNotes | Set-Content -LiteralPath (Join-Path $ReleaseRoot 'RELEASE_NOTES.md') -Encoding utf8

Write-Host "Release packages created in $ReleaseRoot"
$Checksums | ForEach-Object { Write-Host $_ }
