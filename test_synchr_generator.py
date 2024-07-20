import asyncio
import sys


async def async_word_receiver():
    words = ["hello", "world", "this", "is", "an", "async", "generator"]
    for word in words:
        await asyncio.sleep(0.1)  # Simulate a delay in receiving words
        yield word


def word_generator_from_async():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_gen = async_word_receiver()

    async def get_next_word():
        try:
            return await async_gen.__anext__()
        except StopAsyncIteration:
            return None

    while True:
        word = loop.run_until_complete(get_next_word())
        if word is None:
            break
        yield word


# Print words with a typewriter effect
for word in word_generator_from_async():
    sys.stdout.write(word + " ")
    sys.stdout.flush()
