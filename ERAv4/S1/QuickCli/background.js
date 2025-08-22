const aliasToUrl = {
  google:  "https://www.google.com",
  youtube: "https://www.youtube.com",
  github:  "https://github.com",
  docs:    "https://docs.google.com",
  sheets:  "https://sheets.google.com",
  slides:  "https://slides.google.com"
};

function openTab(url) {
  chrome.tabs.create({ url });
}

function isAlias(cmd) {
  return Object.prototype.hasOwnProperty.call(aliasToUrl, cmd);
}

let nativePort = null;

function getNativePort() {
  if (!nativePort) {
    nativePort = chrome.runtime.connectNative("com.quickcli.host");
    nativePort.onDisconnect.addListener(() => {
      console.error("Native host disconnected");
      nativePort = null;
    });
  }
  return nativePort;
}


chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.type !== "runCommand") return;

  const raw = message.command.trim();
  const parts = raw.split(/\s+/);
  const head  = parts[0].toLowerCase();
  const rest  = parts.slice(1).join(" ");

  // 1) google query
  if (head === "google" && rest) {
    openTab(`https://www.google.com/search?q=${encodeURIComponent(rest)}`);
    sendResponse({ ok: true, msg: `Opened Google search for "${rest}"` });
    return;
  }

  // 2) youtube query
  if (head === "youtube" && rest) {
    openTab(`https://www.youtube.com/results?search_query=${encodeURIComponent(rest)}`);
    sendResponse({ ok: true, msg: `Opened YouTube search for "${rest}"` });
    return;
  }

  // 3) simple alias
  if (isAlias(head) && !rest) {
    openTab(aliasToUrl[head]);
    sendResponse({ ok: true, msg: `Opened ${aliasToUrl[head]}` });
    return;
  }

  // 4) raw URL
  if (/^https?:\/\//i.test(raw)) {
    openTab(raw);
    sendResponse({ ok: true, msg: `Opened ${raw}` });
    return;
  }

    // OS command via persistent native host
  const port = getNativePort();
  try {
    port.postMessage({ command: raw });
    port.onMessage.addListener((response) => {
      sendResponse({
        ok: response.ok !== false,
        msg: response.msg || response.result || JSON.stringify(response)
      });
    });
  } catch (e) {
    sendResponse({ ok: false, msg: `Native messaging error: ${e.message}` });
  }
  
  // 5) OS command via native host
  chrome.runtime.sendNativeMessage("com.quickcli.host", { command: raw }, (response) => {
    if (chrome.runtime.lastError) {
      sendResponse({ ok: false, msg: `Native messaging error: ${chrome.runtime.lastError.message}` });
    } else if (!response) {
      sendResponse({ ok: false, msg: "No response from native host" });
    } else {
      sendResponse({
        ok: response.ok !== false,
        msg: response.msg || response.result || JSON.stringify(response)
      });
    }
  });

  // IMPORTANT in MV3: keep channel open
  return true;
});
