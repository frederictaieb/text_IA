// Paramètres à adapter
const SEGMENTS = 50; // nombre de colonnes dans ta heatmap
const DURATION = 10.0; // durée de l'animation en secondes
const WHITE_DURATION = 5.0; // secondes de blanc au début

// Création de la scène
const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Chargement de la texture heatmap
const textureLoader = new THREE.TextureLoader();
textureLoader.load('heatmap.png', (heatmapTexture) => {
  heatmapTexture.minFilter = THREE.LinearFilter;
  heatmapTexture.magFilter = THREE.LinearFilter;
  heatmapTexture.wrapS = THREE.ClampToEdgeWrapping;
  heatmapTexture.wrapT = THREE.ClampToEdgeWrapping;

  // Shader material
  const material = new THREE.ShaderMaterial({
    uniforms: {
      u_heatmap: { value: heatmapTexture },
      u_time: { value: 0.0 },
      u_duration: { value: DURATION },
      u_segments: { value: SEGMENTS },
      u_whiteDuration: { value: WHITE_DURATION }
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragmentShader: `
      uniform sampler2D u_heatmap;
      uniform float u_time;
      uniform float u_duration;
      uniform float u_segments;
      uniform float u_whiteDuration;
      varying vec2 vUv;

      void main() {
        if (u_time < u_whiteDuration) {
          gl_FragColor = vec4(1.0);
          return;
        }
        float revealTime = u_time - u_whiteDuration;
        float revealDuration = u_duration - u_whiteDuration;
        float t = clamp(revealTime / revealDuration, 0.0, 1.0);
        float idx = t * (u_segments - 1.0);
        float idx0 = floor(idx);
        float idx1 = min(idx0 + 1.0, u_segments - 1.0);
        float frac = idx - idx0;
        float u0 = idx0 / (u_segments - 1.0);
        float u1 = idx1 / (u_segments - 1.0);

        vec4 color0 = texture2D(u_heatmap, vec2(u0, vUv.y));
        vec4 color1 = texture2D(u_heatmap, vec2(u1, vUv.y));
        vec4 color = mix(color0, color1, frac);

        gl_FragColor = color;
      }
    `
  });

  // Plan plein écran
  const geometry = new THREE.PlaneGeometry(2, 2);
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  // Animation
  const startTime = performance.now();
  function animate() {
    let t = ((performance.now() - startTime) / 1000.0) % DURATION;
    material.uniforms.u_time.value = t;
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  animate();
});