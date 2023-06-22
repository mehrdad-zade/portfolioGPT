def gpt_res_is_invalid(gptResponse):
    if "september 2021" in gptResponse.lower() or "language model" in gptResponse.lower() or gptResponse == '' or 'cannot provide' in gptResponse:
        return True
    return False