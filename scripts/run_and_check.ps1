# run build/deferred.exe and wait 3 secs.
# check stdout contains word "error" 
# repeat it given number of times and report success or fail
$red = "`e[31m"
$green = "`e[32m"
$reset = "`e[0m"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# pushd $scriptDir/../build
$exePath = Join-Path $scriptDir "..\build\deferred.exe"
$outputFile = Join-Path $scriptDir "output.txt"
$errorFile = Join-Path $scriptDir "error.txt"
$repeatCount = 100
$waitSeconds = 3
$success = $true
for ($i = 1; $i -le $repeatCount; $i++) {
    Write-Host "Run #${i}: Executing $exePath ..."
    # redirect stderr also to the file
    $process = Start-Process -FilePath $exePath -NoNewWindow -PassThru -RedirectStandardOutput $outputFile -RedirectStandardError $errorFile
    Start-Sleep -Seconds $waitSeconds
    $process | Stop-Process
    $output = Get-Content $outputFile
    $error = Get-Content $errorFile
    $output = $output + $error
    # check error word in case insensitive
    if ($output -imatch "error") {
        Write-Host "$red[Error]$reset Run #${i}: Found 'error' in output."
        $success = $false
        break
    } else {
        Write-Host "$green[Pass]$reset Run #${i}: No 'error' found in output."
    }
}

# report final result with stdout in ansi color with separated color variable

if ($success) {
    Write-Host "$green[Success]$reset All $repeatCount runs completed without 'error'."
} else {
    Write-Host "$red[Failed]$reset Some runs contained 'error' in output."
}
# popd