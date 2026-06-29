/**
 * House Price Predictor — Client-Side Prediction Engine
 * =====================================================
 * Loads trained model coefficients from JSON and performs
 * multiple linear regression predictions in the browser.
 */

// ─── Global State ──────────────────────────────────────────────────────────────
let modelData = null;

// Stepper limits
const stepperLimits = {
    bedrooms: { min: 1, max: 6 },
    bathrooms: { min: 1, max: 4 },
    stories: { min: 1, max: 4 },
    parking: { min: 0, max: 3 }
};

// ─── Initialize ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadModel();
    initSliders();
    initSteppers();
    initNavbar();
    initScrollAnimations();
    initPredictButton();
});

// ─── Load Model ────────────────────────────────────────────────────────────────
async function loadModel() {
    try {
        const response = await fetch('model_output.json');
        if (!response.ok) throw new Error('Model file not found');
        modelData = await response.json();
        console.log('✅ Model loaded:', modelData.model_type);
        console.log('   Features:', modelData.feature_columns.length);
        console.log('   R² Score:', modelData.metrics.test_r2);
        updateModelUI();
        predict(); // Initial prediction
    } catch (error) {
        console.error('❌ Error loading model:', error);
        document.getElementById('result-price').textContent = 'Error';
        document.getElementById('result-confidence').innerHTML =
            '<span class="badge-dot" style="background:#ef4444"></span> Model Not Found';
    }
}

// ─── Update Model UI ───────────────────────────────────────────────────────────
function updateModelUI() {
    if (!modelData) return;

    const m = modelData.metrics;

    // Stats
    document.getElementById('stat-r2').textContent = m.test_r2.toFixed(4);
    document.getElementById('stat-rmse').textContent = '$' + formatNumber(m.test_rmse);
    document.getElementById('stat-mae').textContent = '$' + formatNumber(m.test_mae);
    document.getElementById('stat-samples').textContent = m.total_samples;

    // Details
    document.getElementById('detail-features').textContent = modelData.feature_columns.length;
    document.getElementById('detail-train-r2').textContent = m.train_r2.toFixed(4);
    document.getElementById('detail-test-r2').textContent = m.test_r2.toFixed(4);
    document.getElementById('detail-intercept').textContent = '$' + formatNumber(modelData.intercept);
    document.getElementById('detail-avg-price').textContent = '$' + formatNumber(modelData.price_stats.mean);

    // Feature Importance Chart
    renderFeatureChart();

    // Equation
    renderEquation();
}

// ─── Render Feature Importance Chart ───────────────────────────────────────────
function renderFeatureChart() {
    const container = document.getElementById('feature-chart');
    const features = modelData.feature_importance;
    const maxVal = Math.max(...features.map(f => f.abs_coefficient));

    container.innerHTML = features.map(f => {
        const width = Math.max((f.abs_coefficient / maxVal) * 100, 3);
        const sign = f.coefficient >= 0 ? '+' : '';
        return `
            <div class="chart-bar-group">
                <span class="chart-bar-label">${formatFeatureName(f.name)}</span>
                <div class="chart-bar-track">
                    <div class="chart-bar-fill" style="width: 0%;" data-width="${width}%">
                        <span class="chart-bar-value">${sign}${formatNumber(f.coefficient)}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Animate bars after a short delay
    requestAnimationFrame(() => {
        setTimeout(() => {
            container.querySelectorAll('.chart-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width;
            });
        }, 200);
    });
}

// ─── Render Equation ───────────────────────────────────────────────────────────
function renderEquation() {
    const eq = document.getElementById('equation-content');
    const coeffs = modelData.coefficients;
    const intercept = modelData.intercept;

    let equation = `<code><span class="eq-feature">price</span> <span class="eq-op">=</span> <span class="eq-intercept">${formatNumber(intercept)}</span>`;

    for (const [feature, coeff] of Object.entries(coeffs)) {
        const sign = coeff >= 0 ? '+' : '−';
        const absCoeff = Math.abs(coeff);
        equation += `\n  <span class="eq-op">${sign}</span> <span class="eq-coeff">${formatNumber(absCoeff)}</span> <span class="eq-op">×</span> <span class="eq-feature">${feature}</span>`;
    }

    equation += '</code>';
    eq.innerHTML = equation;
}

// ─── Predict ───────────────────────────────────────────────────────────────────
function predict() {
    if (!modelData) return;

    // Gather feature values in the same order as training
    const features = {};

    // Numeric features
    features['area'] = parseInt(document.getElementById('area').value);
    features['bedrooms'] = parseInt(document.getElementById('bedrooms').value);
    features['bathrooms'] = parseInt(document.getElementById('bathrooms').value);
    features['stories'] = parseInt(document.getElementById('stories').value);
    features['parking'] = parseInt(document.getElementById('parking').value);

    // Binary features
    features['mainroad'] = document.getElementById('mainroad').checked ? 1 : 0;
    features['guestroom'] = document.getElementById('guestroom').checked ? 1 : 0;
    features['basement'] = document.getElementById('basement').checked ? 1 : 0;
    features['hotwaterheating'] = document.getElementById('hotwaterheating').checked ? 1 : 0;
    features['airconditioning'] = document.getElementById('airconditioning').checked ? 1 : 0;
    features['prefarea'] = document.getElementById('prefarea').checked ? 1 : 0;

    // One-hot encode furnishing status
    const furnishing = document.querySelector('input[name="furnishing"]:checked').value;
    features['furnishing_furnished'] = furnishing === 'furnished' ? 1 : 0;
    features['furnishing_semi-furnished'] = furnishing === 'semi-furnished' ? 1 : 0;
    features['furnishing_unfurnished'] = furnishing === 'unfurnished' ? 1 : 0;

    // Calculate prediction: price = intercept + Σ(coeff_i × feature_i)
    let price = modelData.intercept;
    for (const col of modelData.feature_columns) {
        const coeff = modelData.coefficients[col];
        const value = features[col] || 0;
        price += coeff * value;
    }

    // Ensure non-negative price
    price = Math.max(0, price);

    // Animate the result
    animatePrice(Math.round(price));
}

// ─── Animate Price ─────────────────────────────────────────────────────────────
function animatePrice(targetPrice) {
    const el = document.getElementById('result-price');
    const resultCard = document.getElementById('result-card');
    const currentPrice = parseInt(el.textContent.replace(/,/g, '')) || 0;
    const diff = targetPrice - currentPrice;
    const duration = 600;
    const startTime = performance.now();

    // Add animation class
    resultCard.classList.add('price-animated');
    setTimeout(() => resultCard.classList.remove('price-animated'), 500);

    function step(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out cubic)
        const eased = 1 - Math.pow(1 - progress, 3);

        const current = Math.round(currentPrice + diff * eased);
        el.textContent = formatNumber(current);

        if (progress < 1) {
            requestAnimationFrame(step);
        }
    }

    requestAnimationFrame(step);
}

// ─── Slider Initialization ─────────────────────────────────────────────────────
function initSliders() {
    const areaSlider = document.getElementById('area');
    const areaValue = document.getElementById('area-value');

    function updateAreaDisplay() {
        areaValue.textContent = formatNumber(parseInt(areaSlider.value));
        updateSliderTrack(areaSlider);
    }

    areaSlider.addEventListener('input', () => {
        updateAreaDisplay();
    });

    updateAreaDisplay();
}

function updateSliderTrack(slider) {
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const val = parseFloat(slider.value);
    const percent = ((val - min) / (max - min)) * 100;
    slider.style.background = `linear-gradient(to right, 
        rgba(99, 102, 241, 0.6) 0%, 
        rgba(139, 92, 246, 0.8) ${percent}%, 
        rgba(99, 102, 241, 0.15) ${percent}%, 
        rgba(99, 102, 241, 0.15) 100%)`;
}

// ─── Stepper Initialization ────────────────────────────────────────────────────
function initSteppers() {
    document.querySelectorAll('.stepper-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.target;
            const input = document.getElementById(target);
            const display = document.getElementById(target + '-value');
            const limits = stepperLimits[target];
            let value = parseInt(input.value);

            if (btn.dataset.action === 'inc') {
                value = Math.min(value + 1, limits.max);
            } else {
                value = Math.max(value - 1, limits.min);
            }

            input.value = value;
            display.textContent = value;

            // Add a brief scale animation
            display.style.transform = 'scale(1.3)';
            setTimeout(() => display.style.transform = 'scale(1)', 150);
        });
    });
}

// ─── Predict Button ────────────────────────────────────────────────────────────
function initPredictButton() {
    const btn = document.getElementById('predict-btn');
    btn.addEventListener('click', (e) => {
        e.preventDefault();
        predict();

        // Scroll to results on mobile
        if (window.innerWidth < 1024) {
            document.getElementById('result-card').scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }
    });

    // Also predict on any input change for real-time updates
    document.querySelectorAll('input[type="range"], input[type="checkbox"], input[name="furnishing"]').forEach(input => {
        input.addEventListener('input', predict);
        input.addEventListener('change', predict);
    });

    document.querySelectorAll('.stepper-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            setTimeout(predict, 50);
        });
    });
}

// ─── Navbar Scroll Effect ──────────────────────────────────────────────────────
function initNavbar() {
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
}

// ─── Scroll Animations ────────────────────────────────────────────────────────
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.animate-in, .glass-card, .about-card').forEach(el => {
        observer.observe(el);
    });
}

// ─── Utilities ─────────────────────────────────────────────────────────────────
function formatNumber(num) {
    return Math.round(num).toLocaleString('en-US');
}

function formatFeatureName(name) {
    const nameMap = {
        'area': 'Area (sq ft)',
        'bedrooms': 'Bedrooms',
        'bathrooms': 'Bathrooms',
        'stories': 'Stories',
        'mainroad': 'Main Road',
        'guestroom': 'Guest Room',
        'basement': 'Basement',
        'hotwaterheating': 'Hot Water Heating',
        'airconditioning': 'Air Conditioning',
        'parking': 'Parking',
        'prefarea': 'Preferred Area',
        'furnishing_furnished': 'Furnished',
        'furnishing_semi-furnished': 'Semi-Furnished',
        'furnishing_unfurnished': 'Unfurnished'
    };
    return nameMap[name] || name;
}
