@echo off
REM Usage example:
REM ./run.bat --experiment_names "imdrf_mapping blurb_genre_collection web_of_science" --model_name_1 t5_baseline --decoder_type_1 transformer --per_level_chars_1 1 --cumulative_1 false --model_name_2 sgt5 --decoder_type_2 transformer --per_level_chars_2 1 --cumulative_2 false --model_name_3 t5_baseline --decoder_type_3 transformer --per_level_chars_3 2 --cumulative_3 true --model_name_4 t5_baseline --decoder_type_4 transformer --per_level_chars_4 2 --cumulative_4 false --model_name_5 sgt5 --decoder_type_5 transformer --per_level_chars_5 1 --cumulative_5 true

setlocal EnableDelayedExpansion

REM === Default fixed parameters ===
set "experiments=blurb_genre_collection"
set "base_data_folder=./data"
set "hf_model_name=google-t5/t5-small"

REM === Parse named arguments as --key value pairs ===
:parse_args
if "%~1"=="" goto done_args

set "arg=%~1"

REM Check for leading --
if "!arg:~0,2!"=="--" (
    set "key=!arg:~2!"
) else (
    echo Error: Expected argument starting with -- but got "!arg!"
    goto done_args
)

shift

if "%~1"=="" (
    echo Error: Missing value for key "!key!"
    goto done_args
)

set "value=%~1"

REM If this is experiment_names, override the default experiments variable
if /I "!key!"=="experiment_names" (
    set "experiments=!value!"
) else (
    REM Otherwise set variable dynamically
    set "!key!=!value!"
)

shift
goto parse_args

:done_args

REM === Loop over possible configurations ===
for %%I in (1 2 3 4 5) do (
    call set "model_name=%%model_name_%%I%%"
    if defined model_name (
        call set "decoder_type=%%decoder_type_%%I%%"
        call set "per_level_chars=%%per_level_chars_%%I%%"
        call set "cumulative=%%cumulative_%%I%%"

        set "cumulative_flag="
        if /I "!cumulative!"=="true" (
            set "cumulative_flag=--cumulative"
        )

        echo ------------------------------------
        echo Running model %%I:
        echo    experiment_names = !experiments!
        echo    model_name      = !model_name!
        echo    decoder_type    = !decoder_type!
        echo    per_level_chars = !per_level_chars!
        echo    cumulative      = !cumulative_flag!
        echo ------------------------------------


        call python .\main.py ^
            --experiment_names !experiments! ^
            --base_data_folder !base_data_folder! ^
            --model_name !model_name! ^
            --decoder_type !decoder_type! ^
            --hf_model_name !hf_model_name! ^
            --per_level_chars !per_level_chars! ^
            !cumulative_flag! | findstr /V "[codecarbon INFO @"
    )
)

endlocal
pause
