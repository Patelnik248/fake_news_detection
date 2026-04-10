document.addEventListener('DOMContentLoaded', () => {
    const newsText = document.getElementById('news-text');
    const currentCount = document.getElementById('current-count');
    const predictBtn = document.getElementById('predict-btn');
    const resultContainer = document.getElementById('result-container');
    const resetBtn = document.getElementById('reset-btn');

    const badge = document.getElementById('result-badge');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceBar = document.getElementById('confidence-bar');
    const resultMessage = document.getElementById('result-message');

    // Character counter
    newsText.addEventListener('input', () => {
        const count = newsText.value.trim().length;
        currentCount.textContent = count;
        predictBtn.disabled = count < 10;
        currentCount.parentElement.classList.toggle('active', count >= 10);
    });

    // Predict
    predictBtn.addEventListener('click', async () => {
        const text = newsText.value.trim();
        if (text.length < 10) return;

        predictBtn.classList.add('loading');
        predictBtn.disabled = true;
        resultContainer.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });

            const data = await response.json();

            if (response.ok) {
                displayResult(data);
            } else {
                alert(data.error || 'Something went wrong. Please try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the server. Is the model trained?');
        } finally {
            predictBtn.classList.remove('loading');
            predictBtn.disabled = false;
        }
    });

    // Reset
    resetBtn.addEventListener('click', () => {
        newsText.value = '';
        currentCount.textContent = '0';
        currentCount.parentElement.classList.remove('active');
        predictBtn.disabled = true;
        resultContainer.classList.add('hidden');
        newsText.focus();
    });

    function displayResult(data) {
        const isReal = data.label === 'Real';
        const confidencePct = (data.confidence * 100).toFixed(1) + '%';
        const type = isReal ? 'real' : 'fake';

        badge.textContent = data.label;
        badge.className = `verdict-badge ${type}`;

        confidenceValue.textContent = confidencePct;

        // Animate bar
        confidenceBar.style.width = '0%';
        confidenceBar.className = `bar-fill ${type}`;
        setTimeout(() => {
            confidenceBar.style.width = confidencePct;
        }, 60);

        resultMessage.innerHTML = `This article shows strong indicators of being <strong>${data.label}</strong> news based on ML analysis.`;

        resultContainer.classList.remove('hidden');
        setTimeout(() => {
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }
});