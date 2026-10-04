param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)
python "$PSScriptRoot\scripts\gary.py" @Args
