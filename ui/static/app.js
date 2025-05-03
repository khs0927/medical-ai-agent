// DOM 요소 선택
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatMessages = document.getElementById('chat-messages');
const useSearchCheckbox = document.getElementById('use-search');
const modelSelect = document.getElementById('model-select');

// 이전 대화 내용을 저장하는 변수
let conversationHistory = [];

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    // 로컬 스토리지에서 설정 불러오기
    if (localStorage.getItem('useSearch') !== null) {
        useSearchCheckbox.checked = localStorage.getItem('useSearch') === 'true';
    }
    if (localStorage.getItem('selectedModel')) {
        modelSelect.value = localStorage.getItem('selectedModel');
    }

    // 폼 제출 이벤트 리스너
    chatForm.addEventListener('submit', handleSubmit);

    // 설정 변경 이벤트 리스너
    useSearchCheckbox.addEventListener('change', () => {
        localStorage.setItem('useSearch', useSearchCheckbox.checked);
    });
    modelSelect.addEventListener('change', () => {
        localStorage.setItem('selectedModel', modelSelect.value);
    });
});

// 폼 제출 처리
async function handleSubmit(e) {
    e.preventDefault();
    
    const message = userInput.value.trim();
    if (!message) return;
    
    // 사용자 메시지 추가
    addMessageToChat('user', message);
    userInput.value = '';
    
    // 입력 필드 비활성화 및 로딩 표시
    userInput.disabled = true;
    chatForm.querySelector('button').disabled = true;
    addTypingIndicator();
    
    try {
        // 서버에 메시지 전송
        const response = await sendMessageToAgent(message);
        // 타이핑 표시 제거
        removeTypingIndicator();
        // 에이전트 응답 추가
        addMessageToChat('agent', response.answer);
    } catch (error) {
        removeTypingIndicator();
        addMessageToChat('system', '오류가 발생했습니다: ' + error.message);
        console.error('Error:', error);
    } finally {
        // 입력 필드 활성화
        userInput.disabled = false;
        chatForm.querySelector('button').disabled = false;
        userInput.focus();
    }
}

// 채팅창에 메시지 추가
function addMessageToChat(sender, message) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    messageElement.textContent = message;
    
    chatMessages.appendChild(messageElement);
    
    // 스크롤을 최하단으로 이동
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // 대화 기록 업데이트
    conversationHistory.push({ role: sender, content: message });
}

// 타이핑 표시 추가
function addTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.classList.add('typing-indicator');
    indicator.id = 'typing-indicator';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('span');
        indicator.appendChild(dot);
    }
    
    chatMessages.appendChild(indicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 타이핑 표시 제거
function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// 에이전트에 메시지 전송
async function sendMessageToAgent(message) {
    const useSearch = useSearchCheckbox.checked;
    const model = modelSelect.value;
    
    const requestData = {
        message: message,
        history: conversationHistory,
        use_search: useSearch,
        model: model
    };
    
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '알 수 없는 오류가 발생했습니다.');
    }
    
    return await response.json();
} 