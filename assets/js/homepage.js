// ===== NAVBAR SCROLL =====
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 50);
}, { passive: true });

// ===== MOBILE NAV =====
const navToggle = document.getElementById('navToggle');
const navLinks  = document.getElementById('navLinks');

navToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
});
navLinks.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => navLinks.classList.remove('open'));
});

// ===== TYPED TEXT =====
const phrases = [
    'AI Infra Engineer',
    'KV Cache Specialist',
    'Search Architect',
    'GPU Vector Search',
    'Agent Memory Builder',
];
let pi = 0, ci = 0, deleting = false;
const typedEl = document.getElementById('typedText');

function type() {
    const word = phrases[pi];
    if (deleting) {
        typedEl.textContent = word.slice(0, --ci);
    } else {
        typedEl.textContent = word.slice(0, ++ci);
    }

    let delay = deleting ? 70 : 110;
    if (!deleting && ci === word.length) { delay = 2200; deleting = true; }
    else if (deleting && ci === 0)       { deleting = false; pi = (pi + 1) % phrases.length; delay = 350; }

    setTimeout(type, delay);
}
setTimeout(type, 1600);

// ===== SCROLL REVEAL =====
const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
        if (entry.isIntersecting) {
            entry.target.style.transitionDelay = `${i * 0.08}s`;
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

// ===== ACTIVE NAV LINK =====
const sections = document.querySelectorAll('section[id]');
const navItems = document.querySelectorAll('.nav-links a');

window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(s => {
        if (window.scrollY >= s.offsetTop - 120) current = s.id;
    });
    navItems.forEach(a => {
        a.classList.toggle('active', a.getAttribute('href') === `#${current}`);
    });
}, { passive: true });
