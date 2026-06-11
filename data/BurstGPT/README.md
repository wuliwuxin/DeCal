https://github.com/HPMLL/BurstGPT

`Timestamp`: request submission time, seconds from 0:00:00 on the first day (calibrated to the local time zone ).

`Session ID`: conversation ID, only conversation mode have this, traces that share the same value of Session ID are in the same conversation session.

`Elapsed time`: time between the request submission time and system response time, covers the full duration from submitting the inference request to getting the complete final response (all tokens included). It’s not the same as TTFT (time to first token). In seconds.

`Model`: called models, including ChatGPT(GPT-3.5) and GPT-4.

`Request tokens`: Request tokens length.

`Response tokens`: Response tokens length.

`Total tokens`: Request tokens length plus response tokens length.

`Log Type`: the way users call the model, in conversation mode or using API, including Conversation log and API log.