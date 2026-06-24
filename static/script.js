// =========================================================================
// ESTADO GLOBAL DE LA APLICACIÓN
// =========================================================================
let playersDatabase = [];
let selectedPlayerA = null;
let selectedPlayerB = null;
let selectedSurface = "Hard";

// Elementos del DOM
const bodyEl = document.body;
const surfaceRadios = document.querySelectorAll('input[name="surface"]');
const surfaceContainers = document.querySelectorAll('.surface-option-btn');

const playerAInput = document.getElementById('player-a-input');
const playerBInput = document.getElementById('player-b-input');
const playerAList = document.getElementById('player-a-list');
const playerBList = document.getElementById('player-b-list');

const quickCardA = document.getElementById('quick-card-a');
const quickCardB = document.getElementById('quick-card-b');

const predictBtn = document.getElementById('predict-btn');
const resultsPanel = document.getElementById('results-panel');

// =========================================================================
// INICIALIZACIÓN Y ENLACE DE EVENTOS
// =========================================================================
document.addEventListener('DOMContentLoaded', () => {
    fetchPlayers();
    setupSurfaceSelector();
    setupAutocomplete(playerAInput, playerAList, 'A');
    setupAutocomplete(playerBInput, playerBList, 'B');
    setupPredictAction();
});

// Obtener los jugadores desde el Backend
async function fetchPlayers() {
    try {
        const response = await fetch('/api/players');
        if (!response.ok) throw new Error('Error al obtener la lista de jugadores');
        playersDatabase = await response.json();
        console.log(`✅ Base de datos cargada: ${playersDatabase.length} jugadores.`);
    } catch (error) {
        console.error('❌ Error cargando base de datos:', error);
    }
}

// Configurar el cambio dinámico de superficies y estilos
function setupSurfaceSelector() {
    surfaceRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            selectedSurface = e.target.value;
            
            // Quitar clase activa previa y asignarla al nuevo contenedor
            surfaceContainers.forEach(container => container.classList.remove('active'));
            radio.parentElement.classList.add('active');
            
            // Cambiar la clase del Body para actualizar las variables CSS
            bodyEl.className = `surface-${selectedSurface.toLowerCase()}`;
            
            // Si ya hay un pronóstico visible, volver a habilitar el botón para recalcular bajo la nueva superficie
            validateInputs();
        });
    });
}

// =========================================================================
// BUSCADOR CON AUTOCOMPLETADO INTERACTIVO
// =========================================================================
function setupAutocomplete(inputEl, listEl, playerType) {
    // Escuchar entrada de teclado
    inputEl.addEventListener('input', (e) => {
        const value = e.target.value.trim().toLowerCase();
        listEl.innerHTML = '';
        
        if (value.length < 2) {
            listEl.style.display = 'none';
            resetPlayerSelection(playerType);
            return;
        }

        // Filtrar coincidencias
        const matches = playersDatabase.filter(player => 
            player.name.toLowerCase().includes(value)
        ).slice(0, 8); // Mostrar máximo 8 sugerencias

        if (matches.length === 0) {
            const noMatch = document.createElement('div');
            noMatch.className = 'autocomplete-item';
            noMatch.innerHTML = `<span class="p-name" style="color: var(--text-muted);">Sin resultados</span>`;
            listEl.appendChild(noMatch);
            listEl.style.display = 'block';
            return;
        }

        // Poblar sugerencias
        matches.forEach(player => {
            const item = document.createElement('div');
            item.className = 'autocomplete-item';
            
            const rankLabel = player.rank && player.rank !== 999 ? `#${player.rank}` : 'S/R';
            
            item.innerHTML = `
                <span class="p-name">${player.name}</span>
                <span class="p-meta">Rank: ${rankLabel} | ELO: ${Math.round(player.elo)}</span>
            `;
            
            item.addEventListener('click', () => {
                inputEl.value = player.name;
                listEl.style.display = 'none';
                selectPlayer(player, playerType);
            });
            
            listEl.appendChild(item);
        });

        listEl.style.display = 'block';
    });

    // Ocultar lista al hacer clic fuera del buscador
    document.addEventListener('click', (e) => {
        if (e.target !== inputEl && e.target !== listEl) {
            listEl.style.display = 'none';
        }
    });

    // Mostrar lista si el input gana foco y ya tiene texto suficiente
    inputEl.addEventListener('focus', () => {
        if (inputEl.value.trim().length >= 2) {
            listEl.style.display = 'block';
        }
    });
}

// Registrar selección del jugador
function selectPlayer(player, playerType) {
    if (playerType === 'A') {
        selectedPlayerA = player;
        updateQuickCard(quickCardA, player, 'A');
    } else {
        selectedPlayerB = player;
        updateQuickCard(quickCardB, player, 'B');
    }
    validateInputs();
}

// Resetear selección
function resetPlayerSelection(playerType) {
    if (playerType === 'A') {
        selectedPlayerA = null;
        resetQuickCard(quickCardA, 'A');
    } else {
        selectedPlayerB = null;
        resetQuickCard(quickCardB, 'B');
    }
    validateInputs();
}

// Actualizar Ficha Rápida
function updateQuickCard(cardEl, player, placeholder) {
    cardEl.classList.add('selected');
    cardEl.querySelector('.avatar').innerText = player.name.charAt(0);
    cardEl.querySelector('.name').innerText = player.name;
    cardEl.querySelector('.rank-val').innerText = player.rank !== 999 ? player.rank : 'Sin Rank (999)';
    cardEl.querySelector('.age-val').innerText = `${player.age} años`;
}

// Limpiar Ficha Rápida
function resetQuickCard(cardEl, placeholder) {
    cardEl.classList.remove('selected');
    cardEl.querySelector('.avatar').innerText = placeholder;
    cardEl.querySelector('.name').innerText = '-';
    cardEl.querySelector('.rank-val').innerText = '-';
    cardEl.querySelector('.age-val').innerText = '-';
}

// Validar estado del formulario para habilitar el botón
function validateInputs() {
    const isPlayerASelected = selectedPlayerA !== null;
    const isPlayerBSelected = selectedPlayerB !== null;
    
    // No permitir seleccionar el mismo jugador para ambos lados
    const areDifferent = isPlayerASelected && isPlayerBSelected && (selectedPlayerA.name !== selectedPlayerB.name);
    
    predictBtn.disabled = !(isPlayerASelected && isPlayerBSelected && areDifferent);
}

// =========================================================================
// ACCIÓN DE PREDICCIÓN Y LLAMADA A LA API
// =========================================================================
function setupPredictAction() {
    predictBtn.addEventListener('click', async () => {
        if (!selectedPlayerA || !selectedPlayerB) return;
        
        // Iniciar estado de carga en el botón
        predictBtn.classList.add('loading');
        predictBtn.disabled = true;
        
        const url = `/api/predict?player_a=${encodeURIComponent(selectedPlayerA.name)}&player_b=${encodeURIComponent(selectedPlayerB.name)}&surface=${selectedSurface}`;
        
        try {
            const response = await fetch(url);
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Fallo en la predicción');
            }
            
            const result = await response.json();
            
            // Simular un retardo cognitivo mínimo (600ms) para mejorar la experiencia de usuario (sensación de cómputo)
            setTimeout(() => {
                renderResults(result);
                predictBtn.classList.remove('loading');
                validateInputs(); // Volver a evaluar el estado
            }, 600);
            
        } catch (error) {
            alert(`Error al realizar la predicción: ${error.message}`);
            predictBtn.classList.remove('loading');
            validateInputs();
        }
    });
}

// Renderizar panel de resultados
function renderResults(res) {
    // Revelar panel quitando la clase hidden
    resultsPanel.classList.remove('hidden');

    // Nombres y porcentajes en la barra
    document.getElementById('prob-name-a').innerText = res.player_a.name;
    document.getElementById('prob-name-b').innerText = res.player_b.name;
    
    const pctA = res.player_a.prob_victory;
    const pctB = res.player_b.prob_victory;
    
    document.getElementById('prob-pct-a').innerText = `${pctA}%`;
    document.getElementById('prob-pct-b').innerText = `${pctB}%`;
    
    // Actualizar barras de probabilidad
    document.querySelector('.prob-bar-a').style.width = `${pctA}%`;
    document.querySelector('.prob-bar-b').style.width = `${pctB}%`;

    // Nombre y confianza del ganador
    const winnerName = res.predicted_winner;
    const isWinnerA = winnerName === res.player_a.name;
    const confidencePct = isWinnerA ? pctA : pctB;
    
    document.getElementById('predicted-winner-display').innerText = winnerName;
    document.getElementById('confidence-display').innerText = `Probabilidad de victoria estimada: ${confidencePct}%`;

    // Detalles Jugador A
    document.getElementById('detail-name-a').innerText = res.player_a.name;
    document.getElementById('detail-elo-hybrid-a').innerText = res.player_a.elo_hybrid.toFixed(1);
    document.getElementById('detail-elo-gen-a').innerText = res.player_a.elo_general.toFixed(1);
    document.getElementById('detail-elo-sup-a').innerText = res.player_a.elo_surface.toFixed(1);
    document.getElementById('detail-rank-a').innerText = res.player_a.rank;
    document.getElementById('detail-age-a').innerText = `${res.player_a.age} años`;

    // Detalles Jugador B
    document.getElementById('detail-name-b').innerText = res.player_b.name;
    document.getElementById('detail-elo-hybrid-b').innerText = res.player_b.elo_hybrid.toFixed(1);
    document.getElementById('detail-elo-gen-b').innerText = res.player_b.elo_general.toFixed(1);
    document.getElementById('detail-elo-sup-b').innerText = res.player_b.elo_surface.toFixed(1);
    document.getElementById('detail-rank-b').innerText = res.player_b.rank;
    document.getElementById('detail-age-b').innerText = `${res.player_b.age} años`;

    // Debugging / Factores Clave Centro
    const diffEloVal = res.features_debug.diff_elo;
    const diffRankVal = res.features_debug.diff_rank;
    const diffAgeVal = res.features_debug.diff_age;
    
    document.getElementById('debug-diff-elo').innerText = `${diffEloVal >= 0 ? '+' : ''}${diffEloVal.toFixed(1)} pts`;
    document.getElementById('debug-diff-rank').innerText = `${diffRankVal >= 0 ? '+' : ''}${diffRankVal} puestos`;
    document.getElementById('debug-diff-age').innerText = `${diffAgeVal >= 0 ? '+' : ''}${diffAgeVal.toFixed(1)} años`;

    // Desplazamiento suave para enfocar los resultados
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
