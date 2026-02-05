# setup-pre-commit.ps1
# PowerShell script to set up pre-commit hooks for the Forex Trading Bot project

Write-Host "Setting up pre-commit hooks for Forex Trading Bot..." -ForegroundColor Green

# Check if pre-commit is installed
$preCommitInstalled = Get-Command pre-commit -ErrorAction SilentlyContinue

if (-not $preCommitInstalled) {
    Write-Host "pre-commit is not installed. Installing..." -ForegroundColor Yellow
    pip install pre-commit
}

# Install the git hooks
Write-Host "Installing git hooks..." -ForegroundColor Green
pre-commit install

# Run on all files to verify setup
Write-Host "Running pre-commit on all files to verify setup..." -ForegroundColor Green
pre-commit run --all-files

Write-Host "Pre-commit hooks have been set up successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "The following hooks are now active:" -ForegroundColor Cyan
Write-Host "- Black: Formats Python code" -ForegroundColor White
Write-Host "- Flake8: Lints Python code" -ForegroundColor White
Write-Host "- Isort: Sorts imports" -ForegroundColor White
Write-Host "- JSON/YAML validation" -ForegroundColor White
Write-Host "- End of file fixer" -ForegroundColor White
Write-Host "- Trailing whitespace removal" -ForegroundColor White
Write-Host "- Debug statement checker" -ForegroundColor White
Write-Host ""
Write-Host "To run pre-commit manually on all files, use:" -ForegroundColor Cyan
Write-Host "  pre-commit run --all-files" -ForegroundColor White
Write-Host ""
Write-Host "To run pre-commit on staged files only, use:" -ForegroundColor Cyan
Write-Host "  pre-commit run" -ForegroundColor White