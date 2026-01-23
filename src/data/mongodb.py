import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


class MongoDBClient:
    def __init__(
        self,
        uri: Optional[str] = None,
        database: str = "EURUSD",
        collection: str = "tf_M1"
    ):
        self.uri = uri or os.getenv("MONGODB_URI")
        self.database_name = database
        self.collection_name = collection
        self._client: Optional[MongoClient] = None

    @property
    def client(self) -> MongoClient:
        if self._client is None:
            self._client = MongoClient(self.uri)
        return self._client

    @property
    def db(self):
        return self.client[self.database_name]

    @property
    def collection(self):
        return self.db[self.collection_name]

    def fetch_ohlc(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLC data for a ticker.

        Args:
            ticker: Currency pair (e.g., 'EURUSD'). Will match both 'EURUSD' and 'C:EURUSD'.
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            limit: Maximum number of records to fetch.

        Returns:
            List of normalized OHLC records with keys: t, o, h, l, c
        """

        query: Dict[str, Any] = {
            "ticker": {"$in": [ticker, f"C:{ticker}"]}
        }

        if start_date or end_date:
            query["t"] = {}
            if start_date:
                query["t"]["$gte"] = datetime.fromisoformat(start_date)
            if end_date:
                query["t"]["$lte"] = datetime.fromisoformat(end_date + "T23:59:59")

        cursor = self.collection.find(
            query,
            {"_id": 0, "t": 1, "o": 1, "h": 1, "l": 1, "c": 1}
        ).sort("t", 1)

        if limit:
            cursor = cursor.limit(limit)

        return list(cursor)

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
