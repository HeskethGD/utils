# MIT License
#
# Copyright (c) [year] [your name]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import random
import time
import openai
import asyncio

# Decorator for async requests


def background(f):
    """Function wrapper for asynchronous execution"""
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

# Decorator for backoffs and retries


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (
        # OLD WAY
#         openai.error.RateLimitError,
#         openai.error.APIError,
#         openai.error.APIConnectionError,
#         openai.error.ServiceUnavailableError
        # NEW WAY
        openai.RateLimitError,
        openai.APIError,
        openai.APIConnectionError,
#         openai.ServiceUnavailableError
        ),
):
    """
    Retry a function with exponential backoff.
    Inspired by OpenAI 
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    """

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                #raise e
                print(str(e))
                return {"error": str(e)}
                
    return wrapper

class AsyncOpenAI:

    def __init__(self, api_key, organization, model="gpt-3.5-turbo", is_in_jupyter=False, sleep=2) -> None:
        # openai api key
        self.api_key = api_key
        # organization name to bill api queries to (optional - will assume default account)
        self.organization = organization
        # openai model
        self.model = model
        # asyncio behaves differently in jupyter
        self.is_in_jupyter = is_in_jupyter
        # random sleep time (s) to stagger api calls (increase this if tasks increase)
        self.sleep = sleep
    
    @retry_with_exponential_backoff
    def openai_api_query(self, messages: list):
        """
        Use OpenAI API to generate a chat completion. 
        Decorated with retries to handle rate limit errors.
        """
        # Set OpenAI API key to the provided token
        openai.api_key = self.api_key
        # Set organization if provided
        if self.organization:
            openai.organization = self.organization
#         OLD WAY
#         response = openai.ChatCompletion.create(
#             model=self.model,
#             messages=messages
#         )
        # NEW WAY - If using the newer package version call it like this
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response 
    
    @background
    def get_openai_response(self, messages: list):
        """Async wrapper for openai call with retries"""
        # Sleep for a random delay to minimise simultaneous api calls and avoid rate limits
        delay_in_seconds = random.random() * self.sleep  
        time.sleep(delay_in_seconds)
        return self.openai_api_query(messages)
    
    async def async_get_openai_responses(self, chats: list):
        """Multiple async calls to openai"""
        futures = [self.get_openai_response(messages) for messages in chats]
        responses = await asyncio.gather(*futures)
        return responses
    
    def get_response(self, chats):
        """
        Main entry for multiple parallel OpenAI calls.
        Chats is an array of OpenAI chats.
        Each chat is an array of messages.
        It may be the case that a chat is just one message with a prompt.
        """
        if self.is_in_jupyter:
            responses = self.async_get_openai_responses(chats)
        else:
            responses = asyncio.run(self.async_get_openai_responses(chats))
        return responses


