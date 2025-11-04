const CONFIG = {
    DEBOUNCE_DELAY: 300,
    MIN_QUERY_LENGTH: 2,
    MAX_SUGGESTIONS: 5,
    SCROLL_BEHAVIOR: 'smooth'
};

const ENDPOINTS = {
    CHAT: '/chat',
    SUGGESTIONS: '/suggestions'
};

const SELECTORS = {
    CHAT_FORM: 'chat-form',
    USER_INPUT: 'user-input',
    CHAT_MESSAGES: 'chat-messages'
};

const chatForm = document.getElementById(SELECTORS.CHAT_FORM);
const userInput = document.getElementById(SELECTORS.USER_INPUT);
const chatMessages = document.getElementById(SELECTORS.CHAT_MESSAGES);

if (!chatForm || !userInput || !chatMessages) {
    console.error('Required DOM elements not found');
    throw new Error('Chat interface could not be initialized');
}

const suggestionsDiv = document.createElement('div');
suggestionsDiv.id = 'suggestions';
suggestionsDiv.className = 'suggestions-dropdown';
chatForm.appendChild(suggestionsDiv);

let debounceTimer = null;
let isSubmitting = false;

async function fetchSuggestions(query) {
    try {
        const response = await fetch(ENDPOINTS.SUGGESTIONS, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data.suggestions || [];
    } catch (error) {
        console.error('Error fetching suggestions:', error);
        return [];
    }
}

function displaySuggestions(suggestions) {
    suggestionsDiv.innerHTML = '';

    if (suggestions.length === 0) {
        suggestionsDiv.style.display = 'none';
        return;
    }

    suggestions.forEach((suggestion, index) => {
        const item = createSuggestionItem(suggestion, index === 0);
        suggestionsDiv.appendChild(item);
    });

    suggestionsDiv.style.display = 'block';
}

function createSuggestionItem(text, isFirst = false) {
    const item = document.createElement('div');
    item.className = 'suggestion-item';
    item.textContent = text;

    item.addEventListener('click', () => {
        selectSuggestion(text);
    });

    item.addEventListener('mouseenter', () => {
        clearActiveSuggestion();
        item.classList.add('suggestion-active');
    });

    return item;
}

function selectSuggestion(text) {
    userInput.value = text;
    hideSuggestions();
    userInput.focus();
}

function clearActiveSuggestion() {
    document.querySelectorAll('.suggestion-item').forEach(item => {
        item.classList.remove('suggestion-active');
    });
}

function hideSuggestions() {
    suggestionsDiv.innerHTML = '';
    suggestionsDiv.style.display = 'none';
}

userInput.addEventListener('input', async (e) => {
    const query = e.target.value.trim();

    clearTimeout(debounceTimer);

    if (query.length < CONFIG.MIN_QUERY_LENGTH) {
        hideSuggestions();
        return;
    }

    debounceTimer = setTimeout(async () => {
        const suggestions = await fetchSuggestions(query);
        displaySuggestions(suggestions);
    }, CONFIG.DEBOUNCE_DELAY);
});

userInput.addEventListener('keydown', (e) => {
    const items = suggestionsDiv.querySelectorAll('.suggestion-item');

    if (items.length === 0) return;

    const activeItem = suggestionsDiv.querySelector('.suggestion-active');

    switch (e.key) {
        case 'ArrowDown':
            e.preventDefault();
            navigateSuggestions(items, activeItem, 'down');
            break;

        case 'ArrowUp':
            e.preventDefault();
            navigateSuggestions(items, activeItem, 'up');
            break;

        case 'Enter':
            if (activeItem) {
                e.preventDefault();
                selectSuggestion(activeItem.textContent);
            }
            break;

        case 'Escape':
            hideSuggestions();
            break;
    }
});

function navigateSuggestions(items, activeItem, direction) {
    const itemsArray = Array.from(items);

    if (!activeItem) {
        const index = direction === 'down' ? 0 : items.length - 1;
        items[index].classList.add('suggestion-active');
        return;
    }

    const currentIndex = itemsArray.indexOf(activeItem);
    activeItem.classList.remove('suggestion-active');

    let nextIndex;
    if (direction === 'down') {
        nextIndex = (currentIndex + 1) % items.length;
    } else {
        nextIndex = (currentIndex - 1 + items.length) % items.length;
    }

    items[nextIndex].classList.add('suggestion-active');
}

document.addEventListener('click', (e) => {
    if (!chatForm.contains(e.target)) {
        hideSuggestions();
    }
});

async function sendMessage(message) {
    try {
        const response = await fetch(ENDPOINTS.CHAT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.status === 'success') {
            return data.response;
        } else {
            throw new Error(data.error || 'Unknown error');
        }
    } catch (error) {
        console.error('Error sending message:', error);
        throw error;
    }
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (isSubmitting) return;

    const message = userInput.value.trim();
    if (!message) return;

    isSubmitting = true;

    hideSuggestions();
    userInput.value = '';
    userInput.disabled = true;

    addMessage(message, 'user');

    const typingDiv = addTypingIndicator();

    try {
        const response = await sendMessage(message);
        typingDiv.remove();
        addMessage(response, 'bot');
    } catch (error) {
        typingDiv.remove();
        addMessage(
            'Sorry, I encountered an error. Please try again or contact rso@berea.edu.',
            'bot'
        );
    } finally {
        isSubmitting = false;
        userInput.disabled = false;
        userInput.focus();
    }

    scrollToBottom();
});

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    const p = document.createElement('p');
    p.textContent = text;

    messageDiv.appendChild(p);
    chatMessages.appendChild(messageDiv);

    scrollToBottom();
}

function addTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message';
    typingDiv.id = 'typing-indicator';

    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';

    typingDiv.appendChild(indicator);
    chatMessages.appendChild(typingDiv);

    scrollToBottom();

    return typingDiv;
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function initializeChat() {
    console.log('RSO Chatbot initialized');

    userInput.focus();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeChat);
} else {
    initializeChat();
}

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});