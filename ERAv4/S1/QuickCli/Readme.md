Quick CLI Browser Extension
    A Chrome extension that provides a terminal-like popup where you can:
    Run web commands/aliases (e.g., google, youtube, github)
    Execute system commands (ls, pwd, cd, etc.) through a native host bridge
    Keep history of last 10 commands
    See the current working directory (cwd) in the prompt

🔧 Features
    google something → opens Google search
    youtube cats → opens YouTube search
    github → opens GitHub directly
    ls, pwd, cd Documents → runs system commands via Python native host
    Command history is saved between sessions (last 10 shown)
    Stays open until you click Exit

📂 Project Structure
    quick-cli-extension/
    │── manifest.json
    │── background.js
    │── popup.html
    │── popup.css
    │── popup.js
    │── native-host/
    │   ├── host.py
    │   ├── com.quickcli.host.json

⚙️ Installation on Chrome
1. Clone or Download
    git clone <repo_url>
    cd quick-cli-extension

2. Enable Developer Mode
    Open chrome://extensions/
    Enable Developer mode (top right)
    Click Load unpacked
    Select the quick-cli-extension/ folder

You should now see Quick CLI in your extensions bar.

🖥️ Native Host Setup (Mac)
System commands (like ls, pwd, cd) need native messaging host.
1. Python Host
Go to native-host/host.py and make sure it has execute permissions:
    chmod +x native-host/host.py

2. Native Host Manifest
Create com.quickcli.host.json:
{
  "name": "com.quickcli.host",
  "description": "Quick CLI Native Messaging Host (Chrome)",
  "path": "/absolute/path/to/native-host/host.py",
  "type": "stdio",
  "allowed_origins": [
    "chrome-extension://<YOUR_EXTENSION_ID>/"
  ]
}
⚠️ Replace <YOUR_EXTENSION_ID> with your Chrome extension ID (see chrome://extensions/).

3. Copy Manifest to Chrome Directory
mkdir -p ~/Library/Application\ Support/Google/Chrome/NativeMessagingHosts/
cp native-host/com.quickcli.host.json ~/Library/Application\ Support/Google/Chrome/NativeMessagingHosts/

▶️ Usage
Click the Quick CLI icon in Chrome toolbar
    Type commands into the prompt:
        $ google openai chatgpt
        $ ls
        $ cd Documents
        $ pwd
    Results show below each command
    Use Exit button to close popup

🛠️ Debugging
1. Check extension logs:
    chrome://extensions/ → Inspect background page
2. Check native host logs (host.py can be patched to log to /tmp/quickcli_host.log):
    tail -f /tmp/quickcli_host.log
3. If you see:
    ✖ Native messaging error: Native host has exited.
        Ensure com.quickcli.host.json points to the correct host.py
        Verify extension ID is correct in JSON
        Ensure host.py runs manually:
            echo -ne '\x2c\x00\x00\x00{"command":"ls"}' | ./native-host/host.py

📌 Notes

Works on Mac; Linux similar (different host directory: ~/.config/google-chrome/NativeMessagingHosts/)
Windows requires .json in %LOCALAPPDATA%\Google\Chrome\User Data\NativeMessagingHosts\ and .bat/.exe wrapper for host.py
Firefox also supported, but needs its own com.quickcli.host.json copied to ~/Library/Application Support/Mozilla/NativeMessagingHosts/

