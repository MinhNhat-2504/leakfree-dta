$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$runOutput = Join-Path $projectRoot 'data\external\chembl_3fam\leakfree\DeepDTA\output-split-cuda'
[void](New-Item -ItemType Directory -Force -Path $runOutput)
$watchLog = Join-Path $runOutput 'watchdog.log'
$completion = Join-Path $runOutput 'COMPLETED.json'
$mutex = New-Object System.Threading.Mutex($false, 'Local\LeakfreeDeepDTAWatchdog')
if (-not $mutex.WaitOne(0)) { exit 0 }
Add-Type @'
using System.Runtime.InteropServices;
public static class DeepDTAPower {
    [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
}
'@
try {
    if (Test-Path $completion) { exit 0 }
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        Add-Content $watchLog "$(Get-Date -Format o) starting attempt $attempt"
        $job = Start-Process wsl.exe -ArgumentList @('-d','Ubuntu','--','bash','/home/minh_nhat/run_deepdta_wsl_gpu.sh') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runOutput "wsl-$attempt.out") -RedirectStandardError (Join-Path $runOutput "wsl-$attempt.err") -PassThru
        while (-not $job.HasExited) {
            [void][DeepDTAPower]::SetThreadExecutionState([uint32]2147483649)
            Start-Sleep -Seconds 20
            $job.Refresh()
        }
        $job.WaitForExit()
        $code = $job.ExitCode
        Add-Content $watchLog "$(Get-Date -Format o) exit code $code"
        if ($code -eq 0) {
            if (-not (Test-Path (Join-Path $runOutput 'predictions.csv'))) { throw 'Process exited without test predictions.' }
            if (-not (Test-Path (Join-Path $runOutput 'run_metadata.json'))) { throw 'Process exited without run metadata.' }
            @{ finished_at = (Get-Date -Format o); status = 'split_runner_completed'; requires_paper_ingestion = $true } | ConvertTo-Json | Set-Content $completion
            $notification = New-Object -ComObject WScript.Shell
            [void]$notification.Popup('DeepDTA upstream da chay xong. Ket qua trong output-split-cuda; can doi chieu va nhap ket qua vao paper.', 0, 'DeepDTA completed', 64)
            exit 0
        }
        Start-Sleep -Seconds 20
    }
    throw 'DeepDTA failed five times; see console.log and wsl-*.err.'
} catch {
    Add-Content $watchLog "$(Get-Date -Format o) ERROR: $_"
    exit 1
} finally {
    [void][DeepDTAPower]::SetThreadExecutionState([uint32]2147483648)
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
