#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from boxsdk import Client, OAuth2
from boxsdk.network.default_network import DefaultNetwork
from boto.dynamodb.condition import NULL
 
class BOXFile():
    
    def __init__(self, CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN):
        # Create OAuth2 object. It's already authenticated, thanks to the developer token.
        self.oauth2 = OAuth2(CLIENT_ID, CLIENT_SECRET, access_token=ACCESS_TOKEN)
        self.client = Client(self.oauth2)
        self._file = None

    def GetUserInfo(self):
        my = self.client.user(user_id='me').get()
        print(my.name.encode("utf-8"))
        print(my.login.encode("utf-8"))
        print(my.avatar_url.encode("utf-8"))
        
    def GetFile(self, fileID):
        self._file = self.client.file(fileID)
        return self._file.content()

