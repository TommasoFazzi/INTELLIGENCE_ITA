// INTEL ITA - Landing Page Interactions

// ============================================
// Particle Background System
// ============================================

class ParticleSystem {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.particleCount = 50;

        this.init();
    }

    init() {
        // Setup canvas
        this.canvas.style.position = 'fixed';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.zIndex = '1';
        this.canvas.style.opacity = '0.3';

        document.querySelector('.hero-bg').appendChild(this.canvas);

        this.resize();
        window.addEventListener('resize', () => this.resize());

        // Create particles
        for (let i = 0; i < this.particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 2 + 1,
                color: Math.random() > 0.5 ? '#FF6B35' : '#00A8E8'
            });
        }

        this.animate();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Update and draw particles
        this.particles.forEach((particle, i) => {
            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;

            // Wrap around edges
            if (particle.x < 0) particle.x = this.canvas.width;
            if (particle.x > this.canvas.width) particle.x = 0;
            if (particle.y < 0) particle.y = this.canvas.height;
            if (particle.y > this.canvas.height) particle.y = 0;

            // Draw particle
            this.ctx.fillStyle = particle.color;
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fill();

            // Draw connections
            this.particles.slice(i + 1).forEach(otherParticle => {
                const dx = particle.x - otherParticle.x;
                const dy = particle.y - otherParticle.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < 150) {
                    this.ctx.strokeStyle = particle.color;
                    this.ctx.globalAlpha = (1 - distance / 150) * 0.3;
                    this.ctx.lineWidth = 0.5;
                    this.ctx.beginPath();
                    this.ctx.moveTo(particle.x, particle.y);
                    this.ctx.lineTo(otherParticle.x, otherParticle.y);
                    this.ctx.stroke();
                    this.ctx.globalAlpha = 1;
                }
            });
        });

        requestAnimationFrame(() => this.animate());
    }
}

// Initialize particle system
new ParticleSystem();

// ============================================
// Interactive Liquid Gradient Effect
// ============================================

const liquidGradient = document.createElement('div');
liquidGradient.className = 'liquid-gradient';
liquidGradient.style.cssText = `
    position: fixed;
    width: 250px;
    height: 250px;
    border-radius: 50%;
    pointer-events: none;
    z-index: 2;
    transform: translate(-50%, -50%);
    transition: opacity 0.5s ease, background 0.3s ease;
    opacity: 0;
    filter: blur(60px);
    mix-blend-mode: screen;
`;
document.body.appendChild(liquidGradient);

let mouseX = 0;
let mouseY = 0;
let currentX = 0;
let currentY = 0;

// Smooth follow animation
function animateLiquid() {
    currentX += (mouseX - currentX) * 0.1;
    currentY += (mouseY - currentY) * 0.1;

    liquidGradient.style.left = currentX + 'px';
    liquidGradient.style.top = currentY + 'px';

    requestAnimationFrame(animateLiquid);
}
animateLiquid();

document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    liquidGradient.style.opacity = '1';

    // Calculate color based on position
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;

    // Normalize position (0-1)
    const normalizedX = e.clientX / windowWidth;
    const normalizedY = e.clientY / windowHeight;

    // Calculate RGB values based on position
    // Left = Orange, Right = Blue, Top = Purple
    const r = Math.floor(255 * (1 - normalizedX) + 107 * normalizedX);
    const g = Math.floor(107 * (1 - normalizedX) + 168 * normalizedX);
    const b = Math.floor(53 * (1 - normalizedX) + 232 * normalizedX);

    // Add purple tint based on Y position
    const purpleInfluence = normalizedY * 0.3;
    const finalR = Math.floor(r + (150 - r) * purpleInfluence);
    const finalG = Math.floor(g + (50 - g) * purpleInfluence);
    const finalB = Math.floor(b + (200 - b) * purpleInfluence);

    // Create dynamic gradient
    liquidGradient.style.background = `
        radial-gradient(
            circle at center,
            rgba(${finalR}, ${finalG}, ${finalB}, 0.25) 0%,
            rgba(${finalR}, ${finalG}, ${finalB}, 0.15) 30%,
            transparent 70%
        )
    `;
});

document.addEventListener('mouseleave', () => {
    liquidGradient.style.opacity = '0';
});

// ============================================
// Smooth scroll for navigation links
// ============================================

// Smooth scroll for navigation links
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

// ============================================
// Navbar scroll effect
// ============================================

// Navbar scroll effect
let lastScroll = 0;
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;

    if (currentScroll > 100) {
        navbar.style.background = 'rgba(10, 22, 40, 0.95)';
        navbar.style.boxShadow = '0 4px 6px -1px rgb(0 0 0 / 0.1)';
    } else {
        navbar.style.background = 'rgba(10, 22, 40, 0.8)';
        navbar.style.boxShadow = 'none';
    }

    lastScroll = currentScroll;
});

// ============================================
// Intersection Observer for fade-in animations
// ============================================

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe feature cards
document.querySelectorAll('.feature-card').forEach((card, index) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(30px)';
    card.style.transition = `all 0.6s ease ${index * 0.1}s`;
    observer.observe(card);
});

// ============================================
// Button ripple effect
// ============================================

// Button ripple effect
document.querySelectorAll('.btn').forEach(button => {
    button.addEventListener('click', function (e) {
        const ripple = document.createElement('span');
        const rect = this.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;

        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        ripple.classList.add('ripple');

        this.appendChild(ripple);

        setTimeout(() => ripple.remove(), 600);
    });
});

// Add ripple styles dynamically
const style = document.createElement('style');
style.textContent = `
    .btn {
        position: relative;
        overflow: hidden;
    }
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: scale(0);
        animation: ripple-animation 0.6s ease-out;
        pointer-events: none;
    }
    @keyframes ripple-animation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ============================================
// Typing effect for hero title (optional)
// ============================================

const heroTitle = document.querySelector('.hero-title');
if (heroTitle) {
    const originalText = heroTitle.innerHTML;
    heroTitle.style.opacity = '0';

    setTimeout(() => {
        heroTitle.style.opacity = '1';
        heroTitle.style.animation = 'fadeInUp 0.8s ease-out';
    }, 300);
}

// Add fadeInUp animation
const fadeInUpStyle = document.createElement('style');
fadeInUpStyle.textContent = `
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(fadeInUpStyle);

// ============================================
// Console easter egg
// ============================================

// Console easter egg
console.log('%c🎯 INTEL ITA', 'font-size: 24px; font-weight: bold; color: #FF6B35;');
console.log('%cIntelligence Platform powered by AI', 'font-size: 14px; color: #94a3b8;');
console.log('%cInterested in joining? Contact us!', 'font-size: 12px; color: #00A8E8;');
