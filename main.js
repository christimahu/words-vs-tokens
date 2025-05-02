// main.js
// =============================================================================
// Drives the Neural Language Space Explorer visualization:
// Fetches data, populates UI (passage text, lists), renders Plotly chart,
// handles theming, includes footer clock. Adheres to separation of concerns.
// Includes safeguard against potential undefined line data during trace creation.
// Increases text label size on plot markers.
// =============================================================================

'use strict';

document.addEventListener('DOMContentLoaded', () => {

    // --- DOM Element References ---
    const chartDiv = document.getElementById('chart');
    const passagePlaceholder = document.getElementById('passage-text-placeholder');
    const wordListUl = document.getElementById('word-list');
    const tokenListUl = document.getElementById('token-list');
    const chartContainer = document.getElementById('chart-container');
    const timeElement = document.getElementById('current-time');

    // --- Configuration ---
    const dataFile = 'data.json';
    const DISTANCE_THRESHOLD = 1.5;
    // Line Styles
    const WORD_LINE_COLOR = 'rgba(200, 100, 200, 0.6)'; const WORD_LINE_DASH = 'dash';
    const TOKEN_LINE_COLOR = 'rgba(180, 180, 180, 0.5)'; const TOKEN_LINE_DASH = 'dot';
    const COMMON_LINE_WIDTH = 1;
    // Plot Text Size
    const WORD_LABEL_SIZE = 12; // Increased size for word labels on plot

    // --- Helper Functions --- (Keep all helper functions as they were)
    const populateList = (ulElement, items) => {
        ulElement.innerHTML = '';
        if (!items || items.length === 0) {
            ulElement.innerHTML = '<li>None found in vocabulary.</li>'; return;
        }
        items.forEach(item => { const li = document.createElement('li'); li.textContent = item; ulElement.appendChild(li); });
    };
    const updatePlotlyLayoutColors = (isDarkMode) => {
        const styles = getComputedStyle(document.documentElement);
        const axisColor = isDarkMode ? styles.getPropertyValue('--color-subheading-dark').trim()||'#a0a0ff' : styles.getPropertyValue('--color-subheading-light').trim()||'#445599';
        const gridColor = isDarkMode ? 'rgba(120, 120, 220, 0.2)' : 'rgba(68, 85, 153, 0.2)';
        const tickColor = isDarkMode ? styles.getPropertyValue('--color-text-secondary-dark').trim()||'#b0b0d0' : styles.getPropertyValue('--color-text-secondary-light').trim()||'#555';
        const legendBgColor = isDarkMode ? styles.getPropertyValue('--color-content-bg-dark').trim()||'rgba(10, 10, 30, 0.7)' : styles.getPropertyValue('--color-content-bg-light').trim()||'rgba(255, 255, 255, 0.7)';
        const legendBorderColor = isDarkMode ? styles.getPropertyValue('--color-border-dark').trim()||'rgba(120, 120, 220, 0.4)' : styles.getPropertyValue('--color-border-light').trim()||'rgba(68, 85, 153, 0.4)';
        const legendFontColor = isDarkMode ? styles.getPropertyValue('--color-text-primary-dark').trim()||'#e0e0e0' : styles.getPropertyValue('--color-text-primary-light').trim()||'#333';
        const axisBgColor = isDarkMode ? 'rgba(10, 10, 30, 0.3)' : 'rgba(230, 230, 240, 0.3)';
        if (chartDiv && chartDiv.layout) {
            Plotly.relayout(chartDiv, {
                'font.color': tickColor,
                'scene.bgcolor': 'rgba(0,0,0,0)',
                'scene.xaxis.color': tickColor, 'scene.yaxis.color': tickColor, 'scene.zaxis.color': tickColor,
                'scene.xaxis.title.font.color': axisColor, 'scene.yaxis.title.font.color': axisColor, 'scene.zaxis.title.font.color': axisColor,
                'scene.xaxis.gridcolor': gridColor, 'scene.yaxis.gridcolor': gridColor, 'scene.zaxis.gridcolor': gridColor,
                'scene.xaxis.backgroundcolor': axisBgColor, 'scene.yaxis.backgroundcolor': axisBgColor, 'scene.zaxis.backgroundcolor': axisBgColor,
                'legend.bgcolor': legendBgColor, 'legend.bordercolor': legendBorderColor, 'legend.font.color': legendFontColor
            });
        }
     };
    const displayError = (message, targetElement) => {
        const isDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches;
        const errorColor = isDark ? '#ff8080' : '#c00000';
        if (targetElement) {
            targetElement.innerHTML = `<p style="text-align: center; padding: 30px; color: ${errorColor};"><strong>Error:</strong> ${message}</p>`;
        } else { console.error("Target element for error display not found:", message); }
     };
    const euclideanDistance3D = (p1, p2) => Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2) + Math.pow(p1.z - p2.z, 2));
    const createLineTraceData = (points, threshold) => {
        const lineX = [], lineY = [], lineZ = [];
        const numPoints = points.length;
        for (let i = 0; i < numPoints; i++) {
            for (let j = i + 1; j < numPoints; j++) {
                const p1 = points[i], p2 = points[j];
                if (euclideanDistance3D(p1, p2) < threshold) {
                    lineX.push(p1.x, p2.x, null); lineY.push(p1.y, p2.y, null); lineZ.push(p1.z, p2.z, null);
                }
            }
        }
        return { lineX, lineY, lineZ };
     };
    const updateTime = () => {
        if (!timeElement) return;
        const now = new Date();
        const options = { timeZone: 'America/Los_Angeles', hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true };
         try { timeElement.textContent = new Intl.DateTimeFormat('en-US', options).format(now); }
         catch (e) { timeElement.textContent = now.toLocaleTimeString(); console.warn("Could not format time with timezone:", e); }
     };

    // --- Main Data Loading and Visualization Logic ---
    const initializeVisualization = async () => {
        try {
            // Step 1 & 2: Fetch and Parse Data
            console.log(`Workspaceing data from: ${dataFile}`);
            const response = await fetch(dataFile);
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status} - Failed to load '${dataFile}'.`);
            console.log("Parsing JSON data...");
            const data = await response.json();
            console.log("Data loaded successfully.");

            // Step 3: Populate UI Elements
            if (passagePlaceholder) { passagePlaceholder.textContent = data.passage || "[Passage text not found.]"; }
            else { console.warn("Passage placeholder not found."); }
            populateList(wordListUl, data.words);
            populateList(tokenListUl, data.tokens);

            // Step 4: Prepare Plotly Data
            if (!data.embeddings || data.embeddings.length === 0) {
                console.warn("No embedding data found.");
                displayError("No embedding data found. Check input or Python script.", chartContainer);
                if (passagePlaceholder) passagePlaceholder.textContent = '[Error loading data]';
                return;
            }
            const wordsData = data.embeddings.filter(d => d.type === 'word');
            const tokensData = data.embeddings.filter(d => d.type === 'token');
            console.log(`Prepared ${wordsData.length} word points and ${tokensData.length} token points.`);

            // Step 5: Generate Line Data
            console.log(`Generating lines for threshold: ${DISTANCE_THRESHOLD}`);
            const wordLineData = createLineTraceData(wordsData, DISTANCE_THRESHOLD);
            const tokenLineData = createLineTraceData(tokensData, DISTANCE_THRESHOLD);
            console.log(`Generated ${wordLineData?.lineX?.length / 3 ?? 0} word lines, ${tokenLineData?.lineX?.length / 3 ?? 0} token lines.`);

            // Step 6: Define Plotly Traces (with Safeguard and Updated Text Size)
            console.log("Defining Plotly traces...");
            const wordLinesTrace = { x: wordLineData ? wordLineData.lineX : [], y: wordLineData ? wordLineData.lineY : [], z: wordLineData ? wordLineData.lineZ : [], mode: 'lines', type: 'scatter3d', name: 'Word Links (Nearby)', line: { color: WORD_LINE_COLOR, width: COMMON_LINE_WIDTH, dash: WORD_LINE_DASH }, hoverinfo: 'none' };
            const tokenLinesTrace = { x: tokenLineData ? tokenLineData.lineX : [], y: tokenLineData ? tokenLineData.lineY : [], z: tokenLineData ? tokenLineData.lineZ : [], mode: 'lines', type: 'scatter3d', name: 'Token Links (Nearby)', line: { color: TOKEN_LINE_COLOR, width: COMMON_LINE_WIDTH, dash: TOKEN_LINE_DASH }, hoverinfo: 'none' };

            const wordTrace = {
                x: wordsData.map(d=>d.x), y: wordsData.map(d=>d.y), z: wordsData.map(d=>d.z),
                mode: 'markers+text', type: 'scatter3d', name: 'Word',
                text: wordsData.map(d=>d.label), textposition: 'top center',
                marker: { color: 'magenta', size: 6, opacity: 0.9 },
                textfont: { color: 'magenta', size: WORD_LABEL_SIZE } // Use constant for size
            };

            const tokenTrace = { x: tokensData.map(d=>d.x), y: tokensData.map(d=>d.y), z: tokensData.map(d=>d.z), mode: 'markers', type: 'scatter3d', name: 'Token (5-char)', text: tokensData.map(d=>d.label), hoverinfo: 'text+name', marker: { color: 'cyan', size: 4, opacity: 0.8 } };
            console.log("Defined traces");

            // Step 7: Define Plotly Layout
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                scene: {
                    bgcolor: 'rgba(0,0,0,0)',
                    xaxis: {title: 'PCA Comp 1', showbackground: true, zeroline: false, showgrid: true},
                    yaxis: {title: 'PCA Comp 2', showbackground: true, zeroline: false, showgrid: true},
                    zaxis: {title: 'PCA Comp 3', showbackground: true, zeroline: false, showgrid: true},
                    camera: {eye: {x: 1.5, y: 1.5, z: 1}}
                },
                font: { family: 'Fira Sans, sans-serif' }, // Base font for layout elements (axes, legend)
                legend: { borderwidth: 1, x: 0.5, y: -0.1, xanchor: 'center', orientation: 'h'},
                margin: { l: 5, r: 5, b: 40, t: 5 }, hovermode: 'closest'
            };

            // Step 8: Render the Chart
            console.log("Rendering Plotly chart...");
            const tracesToPlot = [wordLinesTrace, tokenLinesTrace, wordTrace, tokenTrace];
            await Plotly.newPlot(chartDiv, tracesToPlot, layout, { responsive: true });
            console.log("Chart rendered.");

             // Trigger resize shortly after initial plot
            setTimeout(() => {
                console.log("Triggering post-render resize for Plotly chart.");
                try { Plotly.Plots.resize(chartDiv); }
                catch(resizeError) { console.warn("Error triggering post-render resize:", resizeError); }
             }, 100);

            // Step 9: Apply Theme and Set Up Listeners
            const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches;
            updatePlotlyLayoutColors(prefersDark);
            window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener('change', event => {
                console.log(`Theme changed. Dark: ${event.matches}`);
                updatePlotlyLayoutColors(event.matches);
            });
             window.addEventListener('resize', () => {
                 console.log("Window resized.");
                 const currentPrefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches;
                 updatePlotlyLayoutColors(currentPrefersDark); // Update colors
                 try { Plotly.Plots.resize(chartDiv); console.log("Plotly resize function called on window resize."); } // Trigger Plotly resize
                 catch(resizeError) { console.warn("Error triggering Plotly resize on window resize:", resizeError); }
             });

            // Step 10: Initialize and Start Clock
            if (timeElement) { updateTime(); setInterval(updateTime, 1000); console.log("Footer clock initialized."); }
            else { console.warn("Footer time element (#current-time) not found."); }

        } catch (error) {
            // --- Error Handling ---
            console.error('Initialization failed:', error);
            displayError(`${error.message}. Check console & ensure '${dataFile}' is valid/accessible.`, chartContainer || document.body);
            if (passagePlaceholder) passagePlaceholder.textContent = '[Error loading passage]';
            populateList(wordListUl, []); populateList(tokenListUl, []);
        }
    };

    // --- Initialize ---
    initializeVisualization();

}); // End of DOMContentLoaded listener
