import httpx
import asyncio
import numpy as np

class AsyncRustyDawgClient:
    """Make several requests asynchronously and pool the results."""

    def __init__(self, hosts: list[str] = ["localhost:5000"], read_timeout=60.0):
        self.hosts = hosts
        self.timeout = httpx.Timeout(5.0, read=read_timeout)
    
    async def post_async(self, url, json):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.post(url, json=json)

    async def raw_query(self, json):
        urls = [f"http://{host}/api/cdawg" for host in self.hosts]
        results = await asyncio.gather(*map(lambda url: self.post_async(url, json), urls))
        blobs = [res.json() for res in results]

        all_lengths = []
        all_counts = []

        n_docs = len(json["text"]) if "text" in json else len(json["tokens"])
        for doc in range(n_docs):
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

    async def query(self, json, n_tries: int = 10):
        """Wrapper to avoid rare unpredictable errors. Just try again."""
        try:
            return await self.raw_query(json)
        except (httpx.ReadError, httpx.RemoteProtocolError):
            if n_tries == 0:
                raise RuntimeError("max # of tries exceeded")
            else:
                return self.query(json, n_tries - 1)

async def test_async():
    json = {"text": ["Four score and seven years ago, Rusty DAWG was launched."]}
    client = AsyncRustyDawgClient(["localhost:5000", "localhost:5001"])
    print("Got:", await client.query(json))

if __name__ == "__main__":
    asyncio.run(test_async())