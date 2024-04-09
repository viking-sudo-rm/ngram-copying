import httpx
import asyncio
import numpy as np

async def post_async(url, json):
    async with httpx.AsyncClient() as client:
        return await client.post(url, json=json)

class AsyncRustyDawgClient:
    """Make several requests asynchronously and pool the results."""

    def __init__(self, hosts: list[str] = ["localhost:5000"]):
        self.hosts = hosts
    
    async def query(self, text):
        urls = [f"http://{host}/api/cdawg" for host in self.hosts]
        json  = {"text": text}
        results = await asyncio.gather(*map(lambda url: post_async(url, json), urls))
        blobs = [res.json() for res in results]

        all_lengths = []
        all_counts = []

        for doc in range(len(text)):
            # First aggregate the lengths by taking the max.
            lengths = [blob["lengths"][doc] for blob in blobs]
            lengths = np.stack(lengths, axis=0)  # [n_machines, n_tokens]
            max_lengths = np.max(lengths, axis=0)
            all_lengths.append(max_lengths.tolist())

            counts = [blob["counts"][doc] for blob in blobs]
            counts = np.stack(counts, axis=0)  # [n_machines, n_tokens]
            sum_counts = np.sum(counts * (lengths == max_lengths), axis=0)
            all_counts.append(sum_counts.tolist())

        return {
            "tokens": blobs[0]["tokens"],
            "lengths": all_lengths,
            "counts": all_counts,
        }

async def test_async():
    text = ["Four score and seven years ago, Rusty DAWG was launched."]
    client = AsyncRustyDawgClient(["localhost:5000", "localhost:5001"])
    print("Got:", await client.query(text))

if __name__ == "__main__":
    asyncio.run(test_async())