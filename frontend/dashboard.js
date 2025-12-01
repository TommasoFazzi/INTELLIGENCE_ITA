// Dashboard specific JavaScript

// Simulate live data updates
function updateStats() {
    const statValues = document.querySelectorAll('.stat-value');
    statValues.forEach(stat => {
        const currentValue = parseInt(stat.textContent.replace(/,/g, ''));
        // Random small change
        const change = Math.floor(Math.random() * 3) - 1;
        const newValue = currentValue + change;
        if (newValue > 0) {
            stat.textContent = newValue.toLocaleString();
        }
    });
}

// Update stats every 30 seconds (simulated)
// setInterval(updateStats, 30000);

// Filter functionality
const filterSelect = document.querySelector('.filter-select');
if (filterSelect) {
    filterSelect.addEventListener('change', (e) => {
        const filterValue = e.target.value;
        const reportCards = document.querySelectorAll('.report-card');

        reportCards.forEach(card => {
            const badge = card.querySelector('.report-badge');
            if (filterValue === 'All Types') {
                card.style.display = 'block';
            } else {
                const badgeText = badge.textContent.trim();
                card.style.display = badgeText === filterValue ? 'block' : 'none';
            }
        });
    });
}

// Search functionality
const searchInput = document.querySelector('.search-box input');
if (searchInput) {
    searchInput.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        const reportCards = document.querySelectorAll('.report-card');

        reportCards.forEach(card => {
            const title = card.querySelector('.report-title').textContent.toLowerCase();
            const excerpt = card.querySelector('.report-excerpt').textContent.toLowerCase();

            if (title.includes(searchTerm) || excerpt.includes(searchTerm)) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
    });
}

// Animate stats on page load
window.addEventListener('load', () => {
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';

        setTimeout(() => {
            card.style.transition = 'all 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
});

console.log('📊 Dashboard loaded');
