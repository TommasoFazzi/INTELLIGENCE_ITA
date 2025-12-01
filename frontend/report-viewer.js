// Report Viewer JavaScript

// Print functionality
const printBtn = document.querySelector('.report-actions button[title="Print"]');
if (printBtn) {
    printBtn.addEventListener('click', () => {
        window.print();
    });
}

// Download functionality (simulated)
const downloadBtn = document.querySelector('.report-actions button[title="Download"]');
if (downloadBtn) {
    downloadBtn.addEventListener('click', () => {
        alert('Download functionality will export report as PDF/Markdown');
        // In real implementation: trigger PDF generation or markdown download
    });
}

// Share functionality (simulated)
const shareBtn = document.querySelector('.report-actions button[title="Share"]');
if (shareBtn) {
    shareBtn.addEventListener('click', () => {
        if (navigator.share) {
            navigator.share({
                title: 'Daily Intelligence Briefing',
                text: 'Check out this intelligence report',
                url: window.location.href
            });
        } else {
            // Fallback: copy link to clipboard
            navigator.clipboard.writeText(window.location.href);
            alert('Link copied to clipboard!');
        }
    });
}

// Smooth scroll to sections
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Highlight current section in view (optional)
const sections = document.querySelectorAll('.report-section');
const observerOptions = {
    threshold: 0.3,
    rootMargin: '-100px 0px -50% 0px'
};

const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            // Could highlight corresponding nav item if we had a TOC
            console.log('Viewing section:', entry.target.querySelector('.section-heading')?.textContent);
        }
    });
}, observerOptions);

sections.forEach(section => sectionObserver.observe(section));

// Animate elements on page load
window.addEventListener('load', () => {
    const reportContent = document.querySelector('.report-content');
    const sidebar = document.querySelector('.report-sidebar');

    if (reportContent) {
        reportContent.style.opacity = '0';
        reportContent.style.transform = 'translateY(20px)';

        setTimeout(() => {
            reportContent.style.transition = 'all 0.6s ease';
            reportContent.style.opacity = '1';
            reportContent.style.transform = 'translateY(0)';
        }, 100);
    }

    if (sidebar) {
        sidebar.style.opacity = '0';
        sidebar.style.transform = 'translateX(20px)';

        setTimeout(() => {
            sidebar.style.transition = 'all 0.6s ease';
            sidebar.style.opacity = '1';
            sidebar.style.transform = 'translateX(0)';
        }, 300);
    }
});

console.log('📄 Report viewer loaded');
