const uploadForm = document.querySelector("#upload-form");
const pdfInput = document.querySelector("#pdf-input");
const uploadButton = document.querySelector("#upload-button");
const selectedFileList = document.querySelector("#selected-file-list");
const statusBadge = document.querySelector("#status-badge");
const updatedAt = document.querySelector("#updated-at");
const statusCopy = document.querySelector("#status-copy");
const pdfCount = document.querySelector("#pdf-count");
const chunkCount = document.querySelector("#chunk-count");
const indexedFileList = document.querySelector("#file-list");
const chatStatus = document.querySelector("#chat-status");
const chatHistory = document.querySelector("#chat-history");
const chatForm = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");

let appState = {
    ready: false,
    processing: false,
    indexed_files: [],
    rejected_files: [],
    pdf_count: 0,
    chunk_count: 0,
    status: "Upload one or more PDFs to start building the knowledge base.",
    updated_at: null,
};

let messages = [];
let uploadTimer = null;
let uploadStartedAt = null;

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function renderSelectedFiles() {
    const files = Array.from(pdfInput.files || []);
    selectedFileList.innerHTML = "";

    if (!files.length) {
        const item = document.createElement("li");
        item.textContent = "No files selected.";
        selectedFileList.appendChild(item);
        return;
    }

    files.forEach((file) => {
        const item = document.createElement("li");
        item.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
        selectedFileList.appendChild(item);
    });
}

function renderIndexedFiles() {
    indexedFileList.innerHTML = "";

    if (!appState.indexed_files || !appState.indexed_files.length) {
        const item = document.createElement("li");
        item.textContent = "No PDFs indexed yet.";
        indexedFileList.appendChild(item);
        return;
    }

    appState.indexed_files.forEach((fileName) => {
        const item = document.createElement("li");
        item.textContent = fileName;
        indexedFileList.appendChild(item);
    });

    if (appState.rejected_files && appState.rejected_files.length) {
        const item = document.createElement("li");
        item.textContent = `Ignored: ${appState.rejected_files.join(", ")}`;
        indexedFileList.appendChild(item);
    }
}

function renderMessages() {
    if (!messages.length) {
        chatHistory.innerHTML = `
            <article class="empty-state">
                Upload a PDF, wait for the status to become ready, then start chatting here.
            </article>
        `;
        return;
    }

    chatHistory.innerHTML = messages
        .map(
            (message) => `
                <article class="message ${message.role}">
                    <span class="message-role">${message.role === "user" ? "You" : "Chatbot"}</span>
                    ${escapeHtml(message.text)}
                </article>
            `
        )
        .join("");

    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function renderBadge() {
    let text = "Idle";
    let className = "badge";
    const lowerStatus = (appState.status || "").toLowerCase();

    if (appState.processing) {
        text = "Working";
        className = "badge processing";
    } else if (appState.ready) {
        text = "Ready";
        className = "badge ready";
    } else if (lowerStatus.startsWith("analysis failed")) {
        text = "Error";
        className = "badge error";
    }

    statusBadge.textContent = text;
    statusBadge.className = className;
}

function stopUploadTimer() {
    if (uploadTimer) {
        clearInterval(uploadTimer);
        uploadTimer = null;
    }
    uploadStartedAt = null;
}

function startUploadTimer() {
    stopUploadTimer();
    uploadStartedAt = Date.now();
    uploadTimer = setInterval(() => {
        if (!appState.processing || !uploadStartedAt) {
            stopUploadTimer();
            return;
        }

        const elapsedSeconds = Math.max(1, Math.floor((Date.now() - uploadStartedAt) / 1000));
        statusCopy.textContent = `Analyzing PDFs... ${elapsedSeconds}s elapsed. Keep this tab open until indexing finishes.`;
    }, 1000);
}

function syncControls() {
    uploadButton.disabled = appState.processing || !pdfInput.files.length;

    const chatLocked = appState.processing || !appState.ready;
    messageInput.disabled = chatLocked;
    sendButton.disabled = chatLocked;
    chatStatus.textContent = chatLocked
        ? (appState.processing ? "Analysis is running" : "Chat is locked until indexing is complete")
        : "Knowledge base is ready";
}

function renderState() {
    renderBadge();
    updatedAt.textContent = appState.updated_at || "Not indexed yet";
    statusCopy.textContent = appState.status || "Waiting for upload.";
    pdfCount.textContent = appState.pdf_count || 0;
    chunkCount.textContent = appState.chunk_count || 0;
    renderIndexedFiles();
    syncControls();
}

function applyState(nextState, nextMessages) {
    appState = nextState || appState;
    messages = nextMessages ?? messages;

    if (!appState.processing) {
        stopUploadTimer();
    }

    if (!appState.processing && appState.ready) {
        pdfInput.value = "";
    }

    renderSelectedFiles();
    renderState();
    renderMessages();
}

async function readApiResponse(response) {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
        return response.json();
    }

    const text = await response.text();
    return { error: text || "Unexpected server response." };
}

async function refreshState() {
    try {
        const response = await fetch("/api/state", { cache: "no-store" });
        const payload = await readApiResponse(response);
        applyState(payload.state, payload.messages || []);
    } catch (error) {
        appState.status = "Could not refresh status from the server.";
        renderState();
    }
}

async function uploadPdfs(event) {
    event.preventDefault();

    if (!pdfInput.files.length) {
        appState.status = "Choose at least one PDF before clicking Analyze PDFs.";
        renderState();
        return;
    }

    const formData = new FormData();
    Array.from(pdfInput.files).forEach((file) => formData.append("pdfs", file));

    applyState(
        {
            ...appState,
            ready: false,
            processing: true,
            indexed_files: [],
            rejected_files: [],
            pdf_count: 0,
            chunk_count: 0,
            status: "Analyzing PDFs. This can take up to about 1 minute for larger files.",
            updated_at: null,
        },
        []
    );
    startUploadTimer();

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData,
        });
        const payload = await readApiResponse(response);
        if (!response.ok) {
            if (payload.state) {
                applyState(payload.state, payload.messages || []);
            }
            throw new Error(payload.error || "Upload failed.");
        }

        applyState(payload.state, payload.messages || []);
    } catch (error) {
        stopUploadTimer();
        applyState(
            {
                ...appState,
                ready: false,
                processing: false,
                status: error.message,
            },
            []
        );
    }
}

async function sendMessage(event) {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (!message) {
        return;
    }

    sendButton.disabled = true;
    messageInput.disabled = true;

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message }),
        });
        const payload = await readApiResponse(response);
        if (!response.ok) {
            throw new Error(payload.error || "Chat request failed.");
        }

        messages = payload.messages || messages;
        messageInput.value = "";
        renderMessages();
    } catch (error) {
        appState.status = error.message;
        renderState();
    } finally {
        syncControls();
        messageInput.focus();
    }
}

uploadForm.addEventListener("submit", uploadPdfs);
pdfInput.addEventListener("change", () => {
    renderSelectedFiles();
    syncControls();
});
chatForm.addEventListener("submit", sendMessage);
window.addEventListener("DOMContentLoaded", refreshState);
