if (NOT DEFINED SOURCE_DIR OR NOT DEFINED PATCH_FILE)
    message(FATAL_ERROR "SOURCE_DIR and PATCH_FILE are required")
endif ()

execute_process(
    COMMAND git apply --recount --check "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE apply_check
    OUTPUT_VARIABLE apply_stdout
    ERROR_VARIABLE apply_stderr)

if (apply_check EQUAL 0)
    execute_process(
        COMMAND git apply --recount "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE apply_result
        OUTPUT_VARIABLE apply_output
        ERROR_VARIABLE apply_error)
    if (NOT apply_result EQUAL 0)
        message(FATAL_ERROR "Failed to apply DeepEP patch:\n${apply_output}\n${apply_error}")
    endif ()
else ()
    execute_process(
        COMMAND git apply --recount --reverse --check "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE reverse_check
        OUTPUT_QUIET
        ERROR_QUIET)
    if (NOT reverse_check EQUAL 0)
        message(FATAL_ERROR
            "DeepEP ${SOURCE_DIR} is incompatible with the TurboMind patch.\n"
            "${apply_stdout}\n${apply_stderr}")
    endif ()
endif ()
