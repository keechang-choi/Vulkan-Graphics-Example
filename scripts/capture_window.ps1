<#
.SYNOPSIS
  Launch a Vulkan example, wait for the window, capture it, then kill the process.

.EXAMPLE
  .\scripts\capture_window.ps1 -Exe build\soap_bubble.exe
  .\scripts\capture_window.ps1 -Exe build\pbr.exe -WinX 200 -WinY 50 -OutputPath out.png
#>
param(
    [Parameter(Mandatory = $true)] [string] $Exe,
    [string] $WorkingDirectory,
    [string] $OutputPath,
    [int] $WinX = 100,
    [int] $WinY = 100,
    [int] $WaitSeconds = 5,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'

$Exe = (Resolve-Path $Exe).Path
if (-not $WorkingDirectory) { $WorkingDirectory = Split-Path $Exe }
if (-not $OutputPath) {
    $repoRoot = Split-Path $PSScriptRoot
    $name = [System.IO.Path]::GetFileNameWithoutExtension($Exe)
    $OutputPath = Join-Path $repoRoot "screenshots\$name.png"
}
$outDir = Split-Path $OutputPath
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class CaptureWin {
    [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr hwnd, int dwAttribute, out RECT pvAttribute, int cbAttribute);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr hWnd);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

$args = @('--win-x', "$WinX", '--win-y', "$WinY") + $ExtraArgs
$proc = Start-Process -FilePath $Exe -WorkingDirectory $WorkingDirectory -ArgumentList $args -PassThru
Write-Host "PID: $($proc.Id)  args: $($args -join ' ')"

$hwnd = [IntPtr]::Zero
$deadline = (Get-Date).AddSeconds(15)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 250
    $proc.Refresh()
    if ($proc.HasExited) { throw "Process exited early. Code: $($proc.ExitCode)" }
    if ($proc.MainWindowHandle -ne [IntPtr]::Zero) { $hwnd = $proc.MainWindowHandle; break }
}
if ($hwnd -eq [IntPtr]::Zero) {
    Stop-Process -Id $proc.Id -Force
    throw "Window did not appear within 15s"
}

# SW_RESTORE = 9
[CaptureWin]::ShowWindow($hwnd, 9) | Out-Null
[CaptureWin]::BringWindowToTop($hwnd) | Out-Null
[CaptureWin]::SetForegroundWindow($hwnd) | Out-Null

Start-Sleep -Seconds $WaitSeconds
[CaptureWin]::SetForegroundWindow($hwnd) | Out-Null
Start-Sleep -Milliseconds 500

# DWMWA_EXTENDED_FRAME_BOUNDS = 9 (excludes invisible drop shadow)
$rect = New-Object CaptureWin+RECT
[CaptureWin]::DwmGetWindowAttribute($hwnd, 9, [ref] $rect, 16) | Out-Null
$w = $rect.Right - $rect.Left
$h = $rect.Bottom - $rect.Top
Write-Host "Visible rect: ($($rect.Left),$($rect.Top)) ${w}x${h}"

try {
    $bmp = New-Object System.Drawing.Bitmap $w, $h
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($rect.Left, $rect.Top, 0, 0, (New-Object System.Drawing.Size $w, $h))
    $bmp.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Host "Saved $OutputPath"
}
finally {
    if ($g)   { $g.Dispose() }
    if ($bmp) { $bmp.Dispose() }
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
