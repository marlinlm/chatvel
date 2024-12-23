from typing import Callable, Iterator, List, Optional
import uuid
from langchain_community.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter

from utils.general_utils import async_run

class DataSaver(BaseLoader):
    def __init__(self):
        pass
        
    def load_and_split(
        self, 
    ) -> List[Document]:
        docs = self.load()
        return docs

class DataLoader(BaseLoader):
    def __init__(self,
                 split:bool = False):
        self._split:bool = split
    
    def load_and_split(
        self, 
        text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        if self._split:
            docs = super().load_and_split(text_splitter=text_splitter)
        docs = self.load()
        return docs
    
    def lazy_load(self) -> Iterator[Document]:
        savers = self.on_load_data()
        if savers is None:
            return
        
        for saver in savers:
            docs = saver.load_and_split()
            if docs is None:
                continue
            for doc in docs:
                yield doc

    def on_load_data(self) -> Iterator[DataSaver]:
        raise Exception(f"{self.__class__.__name__} does not implement on_load_data()")
                
    
class DatasetLoader(BaseLoader):
    def __init__(
        self
    ) -> None:
        pass

    def load_and_split(
        self
    ) -> List[Document]:
        return self.load()
    
    def load_data(self) -> Iterator[BaseLoader]:
        raise Exception(f"{self.__class__.__name__} does not implement load_data()")
    
    def lazy_load(self) -> Iterator[Document]:
        loaders = self.load_data()
        if loaders is None:
            return
        
        for loader in loaders:
            docs = loader.load_and_split()
            if docs is None:
                continue
            for doc in docs:
                yield doc

class ChatvelDataSaver(DataSaver):
    def __init__(self,
                data,
                handler:Callable
    ):
        self._handler:Callable = handler
        self._data = data
        
    def lazy_load(self) -> Iterator[Document]:
        docs = self._handler(self._data)
        if docs is None:
            return
        for doc in docs:
            yield doc

class ChatvelDataLoader(DataLoader):
    def __init__(self,
                data,
                handlers:List[Callable] = [],
                ):
        super().__init__(split=False)
        self._data = data 
        self._handlers:List[Callable] = handlers
        
    def on_load_data(self) -> Iterator[DataSaver]:
        for handler in self._handlers:
            yield ChatvelDataSaver(
                data=self._data,
                handler=handler
                )

class ChatvelLoader(DatasetLoader):
    def __init__(self):
        self.id = uuid.uuid4().hex
        
    def on_load_dataset(self) -> Iterator:
        raise Exception(f"{self.__class__.__name__} does not implement on_load_dataset()")
    
    def data_handlers(self) -> List[Callable]:
        raise Exception(f"{self.__class__.__name__} does not implement data_handlers()")
    
    def load_data(self) -> Iterator[ChatvelDataLoader]:
        data_set = self.on_load_dataset()
        if data_set is None:
            return
        for data in data_set:
            yield ChatvelDataLoader(
                        data = data,
                        handlers = self.data_handlers()
                        )