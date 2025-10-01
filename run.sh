#!/bin/bash
# Usage example:
# ./run.sh --experiment_names "blurb_genre_collection other_experiment" --model_name_1 t5_baseline --decoder_type_1 transformer --per_level_chars_1 1 --cumulative_1 false ...

# === Default fixed parameters ===
experiments="blurb_genre_collection"
base_data_folder="./data"
hf_model_name="google-t5/t5-small"

# === Parse named arguments as --key value pairs ===
while [[ $# -gt 0 ]]; do
    arg="$1"
    if [[ "$arg" == --* ]]; then
        key="${arg:2}"
        shift
        if [[ $# -eq 0 ]]; then
            echo "Error: Missing value for key $key"
            exit 1
        fi
        value="$1"
        shift
        if [[ "$key" == "experiment_names" ]]; then
            experiments="$value"
        else
            declare "$key"="$value"
        fi
    else
        echo "Error: Expected argument starting with -- but got $arg"
        exit 1
    fi
done

# === Loop over possible configurations ===
for I in 1 2 3 4; do
    model_name_var="model_name_$I"
    decoder_type_var="decoder_type_$I"
    per_level_chars_var="per_level_chars_$I"
    cumulative_var="cumulative_$I"

    model_name="${!model_name_var}"
    if [[ -n "$model_name" ]]; then
        decoder_type="${!decoder_type_var}"
        per_level_chars="${!per_level_chars_var}"
        cumulative="${!cumulative_var}"

        cumulative_flag=""
        if [[ "$cumulative" == "true" ]]; then
            cumulative_flag="--cumulative"
        fi

        echo "------------------------------------"
        echo "Running model $I:"
        echo "   experiment_names = $experiments"
        echo "   model_name      = $model_name"
        echo "   decoder_type    = $decoder_type"
        echo "   per_level_chars = $per_level_chars"
        echo "   cumulative      = $cumulative_flag"
        echo "------------------------------------"

        python ./main.py \
            --experiment_names "$experiments" \
            --base_data_folder "$base_data_folder" \
            --model_name "$model_name" \
            --decoder_type "$decoder_type" \
            --hf_model_name "$hf_model_name" \
            --per_level_chars "$per_level_chars" \
            $cumulative_flag | grep -v "\[codecarbon INFO @"
    fi
done