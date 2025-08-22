mkdir native-host

touch manifest.json            # extension manifest
touch popup.html               # popup UI (textbox, buttons, 
touch popup.css                # retro terminal styling
touch popup.js                 # popup logic (history, sending commands)
touch background.js            # background script (handles commands)
touch native-host/host.py 
touch native-host/com.quickcli.host.json  # native host manifest (installed on OS)
touch native-host/com.quickcli.host.chrome.json
touch native-host/com.quickcli.host.firefox.json