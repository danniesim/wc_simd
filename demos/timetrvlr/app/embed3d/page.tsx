"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

// Minimal .npy loader supporting v1.x/v2.x float arrays, C-order
function parseNpy(buffer: ArrayBuffer): {
  data: Float32Array;
  shape: number[];
} {
  const magic = new Uint8Array(buffer, 0, 6);
  if (
    !(
      magic[0] === 0x93 &&
      magic[1] === 78 &&
      magic[2] === 85 &&
      magic[3] === 77 &&
      magic[4] === 80 &&
      magic[5] === 89
    )
  ) {
    throw new Error("Invalid NPY file: missing magic header");
  }

  const view = new DataView(buffer);
  const major = view.getUint8(6);
  const minor = view.getUint8(7);

  let headerLen: number;
  let headerOffset: number;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    headerOffset = 10;
  } else if (major === 2 || major === 3) {
    headerLen = view.getUint32(8, true);
    headerOffset = 12;
  } else {
    throw new Error(`Unsupported NPY version ${major}.${minor}`);
  }

  const headerBytes = new Uint8Array(buffer, headerOffset, headerLen);
  const header = new TextDecoder("latin1").decode(headerBytes);

  // Header is a Python dict string, e.g.:
  // {'descr': '<f4', 'fortran_order': False, 'shape': (100, 3), }
  const descrMatch = header.match(/'descr'\s*:\s*'([^']+)'/);
  const shapeMatch = header.match(/'shape'\s*:\s*\(([^\)]*)\)/);
  const fortranMatch = header.match(/'fortran_order'\s*:\s*(True|False)/);

  if (!descrMatch || !shapeMatch || !fortranMatch) {
    throw new Error("Malformed NPY header");
  }

  const descr = descrMatch[1];
  const fortran = fortranMatch[1] === "True";
  if (fortran) {
    throw new Error("Fortran-order arrays are not supported");
  }

  const shape = shapeMatch[1]
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10));

  // Determine typed array
  const littleEndian = descr.startsWith("<") || descr.startsWith("|");
  const type = descr.slice(1); // e.g., f4, f8

  const dataOffset = headerOffset + headerLen;
  const byteOffset =
    dataOffset + (dataOffset % 16 === 0 ? 0 : 16 - (dataOffset % 16)); // align if needed
  const length = buffer.byteLength - byteOffset;

  let floatData: Float32Array;
  if (type === "f4") {
    // Float32
    if (!littleEndian) throw new Error("Big-endian not supported");
    floatData = new Float32Array(buffer, byteOffset, length / 4);
  } else if (type === "f8") {
    // Float64 -> convert to Float32
    if (!littleEndian) throw new Error("Big-endian not supported");
    const f64 = new Float64Array(buffer, byteOffset, length / 8);
    floatData = new Float32Array(f64.length);
    for (let i = 0; i < f64.length; i++) floatData[i] = f64[i];
  } else if (
    type === "i4" ||
    type === "i8" ||
    type === "u4" ||
    type === "u8" ||
    type === "i2" ||
    type === "u2" ||
    type === "i1" ||
    type === "u1"
  ) {
    // Convert ints to float32 for plotting
    let arr: Float32Array;
    if (type === "i4" || type === "u4") {
      const dv = new DataView(buffer, byteOffset, length);
      const n = length / 4;
      arr = new Float32Array(n);
      for (let i = 0; i < n; i++) arr[i] = dv.getInt32(i * 4, true);
    } else if (type === "i8" || type === "u8") {
      const dv = new DataView(buffer, byteOffset, length);
      const n = length / 8;
      arr = new Float32Array(n);
      for (let i = 0; i < n; i++) arr[i] = Number(dv.getBigInt64(i * 8, true));
    } else if (type === "i2" || type === "u2") {
      const dv = new DataView(buffer, byteOffset, length);
      const n = length / 2;
      arr = new Float32Array(n);
      for (let i = 0; i < n; i++) arr[i] = dv.getInt16(i * 2, true);
    } else {
      const dv = new DataView(buffer, byteOffset, length);
      const n = length;
      arr = new Float32Array(n);
      for (let i = 0; i < n; i++) arr[i] = dv.getInt8(i);
    }
    floatData = arr;
  } else {
    throw new Error(`Unsupported dtype: ${descr}`);
  }

  return { data: floatData, shape };
}

export default function Embed3DPage() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [status, setStatus] = useState<string>("Loading...");
  const [error, setError] = useState<string | null>(null);
  const [zClip, setZClip] = useState<number>(6e-2);
  const [logZ, setLogZ] = useState<number>(Math.log10(6e-2));
  const imageIdsRef = useRef<string[] | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const zClipRef = useRef<number>(6e-2);

  useEffect(() => {
    let renderer: THREE.WebGLRenderer | null = null;
    let camera: THREE.PerspectiveCamera | null = null;
    let scene: THREE.Scene | null = null;
    let animationId: number | null = null;
    let disposed = false;

    async function init() {
      setStatus("Fetching NPY and index...");
      try {
        const res = await fetch(
          "/iiif_no_text_embedding_matrix_vlm_embed_vae3d_hires_1.npy"
        );
        if (!res.ok) throw new Error(`Failed to fetch NPY: ${res.status}`);
        const buf = await res.arrayBuffer();
        const { data, shape } = parseNpy(buf);

        // Expect [N,3] or [3,N]
        let n: number;
        let coords: Float32Array;
        if (shape.length === 2 && shape[1] === 3) {
          n = shape[0];
          coords = data;
        } else if (shape.length === 2 && shape[0] === 3) {
          n = shape[1];
          // transpose to Nx3
          coords = new Float32Array(n * 3);
          for (let i = 0; i < n; i++) {
            coords[i * 3 + 0] = data[0 * n + i];
            coords[i * 3 + 1] = data[1 * n + i];
            coords[i * 3 + 2] = data[2 * n + i];
          }
        } else if (shape.length === 1 && shape[0] % 3 === 0) {
          n = shape[0] / 3;
          coords = data;
        } else {
          throw new Error(
            `Unexpected shape ${JSON.stringify(shape)}; expected [N,3]`
          );
        }

        // Load image index JSON from public (frontend-only)
        setStatus("Fetching index JSON...");
        const idxRes = await fetch("/iiif_no_text_embedding_index.json");
        if (!idxRes.ok) {
          throw new Error(`Failed to load index JSON: ${idxRes.status}`);
        }
        const raw = await idxRes.json();
        let imageIds: string[] = [];
        if (Array.isArray(raw)) {
          imageIds = raw.map((v) =>
            v && typeof v === "object" && "image_id" in v
              ? String((v as any).image_id)
              : String(v ?? "")
          );
        } else if (raw && typeof raw === "object") {
          const keys = Object.keys(raw)
            .map((k) => Number(k))
            .filter((n) => Number.isFinite(n) && n >= 0)
            .sort((a, b) => a - b);
          const last = keys.length ? keys[keys.length - 1] : -1;
          imageIds = new Array(last + 1).fill("");
          for (const k of keys) {
            const v = (raw as any)[k];
            const id = v && typeof v === "object" && "image_id" in v ? String(v.image_id) : String(v ?? "");
            imageIds[k] = id;
          }
        } else {
          throw new Error("Unsupported index JSON format");
        }
        imageIdsRef.current = imageIds;

        // Warn on row counts mismatch but continue rendering
        if (imageIds.length !== n) {
          console.warn(
            `[embed3d] Row count mismatch: indexJSON=${imageIds.length} vs numpy=${n}`
          );
        }
        console.log(`[embed3d] points: ${n}; mapped image_ids: ${imageIds.length}`);

        // Center and scale to unit cube
        const mean = [0, 0, 0];
        for (let i = 0; i < n; i++) {
          mean[0] += coords[i * 3 + 0];
          mean[1] += coords[i * 3 + 1];
          mean[2] += coords[i * 3 + 2];
        }
        mean[0] /= n;
        mean[1] /= n;
        mean[2] /= n;
        let maxAbs = 1e-6;
        for (let i = 0; i < n; i++) {
          coords[i * 3 + 0] -= mean[0];
          coords[i * 3 + 1] -= mean[1];
          coords[i * 3 + 2] -= mean[2];
          maxAbs = Math.max(
            maxAbs,
            Math.abs(coords[i * 3 + 0]),
            Math.abs(coords[i * 3 + 1]),
            Math.abs(coords[i * 3 + 2])
          );
        }
        const s = 1.0 / maxAbs;
        for (let i = 0; i < n; i++) {
          coords[i * 3 + 0] *= s;
          coords[i * 3 + 1] *= s;
          coords[i * 3 + 2] *= s;
        }

        setStatus(`Rendering ${n} points...`);

        const {
          WebGLRenderer,
          PerspectiveCamera,
          Scene,
          Color,
          AmbientLight,
          Fog,
          Vector3,
          Quaternion,
          SRGBColorSpace,
          PlaneGeometry,
          MeshBasicMaterial,
          Mesh,
          DoubleSide,
          Group,
          TextureLoader,
          MathUtils,
        } = await import("three");

        if (!containerRef.current) return;

        const width = containerRef.current.clientWidth;
        const height = containerRef.current.clientHeight;

        renderer = new WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        if ("outputColorSpace" in renderer) {
          renderer.outputColorSpace = SRGBColorSpace;
        }
        containerRef.current.appendChild(renderer.domElement);

        scene = new Scene();
        scene.background = new Color(0x000000);
        // Linear fog that corresponds with far clip distance
        scene.fog = new Fog(0x000000, Math.max(0, zClip * 0.6), zClip);

        const cameraDistance = 0; // start at center of the field
        const nearPlane = Math.max(1e-6, Math.min(0.01, zClip * 0.1));
        camera = new PerspectiveCamera(60, width / height, nearPlane, zClip);
        camera.position.set(0, 0, cameraDistance);
        camera.updateProjectionMatrix();

        // Expose for reactive updates
        cameraRef.current = camera;
        sceneRef.current = scene;

        // Ambient light for any future meshes
        scene.add(new AmbientLight(0xffffff, 0.5));

        // Create billboarding planes for each point; textures load lazily
        const planeSize = 0.008; // world units (height) — 10x smaller
        const planeGeom = new PlaneGeometry(1, 1);
        const baseMaterial = new MeshBasicMaterial({
          color: 0x66ccff,
          transparent: true,
          opacity: 0.9,
          side: DoubleSide,
          depthWrite: false,
        });
        const planeGroup = new Group();
        scene.add(planeGroup);

        type PlaneItem = {
          mesh: Mesh;
          status: "empty" | "loading" | "ready";
          texture?: THREE.Texture;
        };
        const planes: PlaneItem[] = [];
        for (let i = 0; i < n; i++) {
          const mesh = new Mesh(planeGeom, baseMaterial.clone());
          mesh.position.set(
            coords[i * 3 + 0],
            coords[i * 3 + 1],
            coords[i * 3 + 2]
          );
          mesh.scale.set(planeSize, planeSize, 1);
          planeGroup.add(mesh);
          planes.push({ mesh, status: "empty" });
        }

        const loader = new TextureLoader();
        (loader as any).setCrossOrigin?.("anonymous");
        const maxConcurrent = 4;
        let inFlight = 0;
        // Scale threshold with planeSize so smaller planes still trigger loads
        const thresholdFrac = 0.05 * (0.008 / 0.08); // effectively 0.005 for planeSize=0.008
        const imageUrls = imageIdsRef.current || [];
        function maybeLoadTexture(pi: PlaneItem, idx: number, canvasH: number) {
          if (!imageUrls || !imageUrls[idx]) return;
          if (pi.status !== "empty" || inFlight >= maxConcurrent) return;
          const url = imageUrls[idx];
          if (!url) return;
          pi.status = "loading";
          inFlight++;
          loader.load(
            url,
            (tex) => {
              try {
                pi.texture = tex;
                (tex as any).colorSpace = SRGBColorSpace;
                const mat = pi.mesh.material as MeshBasicMaterial;
                mat.map = tex as any;
                mat.color.set(0xffffff);
                mat.needsUpdate = true;
                // Maintain aspect ratio: height = planeSize
                const iw = (tex as any).image?.width || 1;
                const ih = (tex as any).image?.height || 1;
                const aspect = iw / Math.max(ih, 1);
                pi.mesh.scale.set(planeSize * aspect, planeSize, 1);
                pi.status = "ready";
              } catch (e) {
                console.error("apply texture failed", e);
                pi.status = "empty";
                try {
                  (tex as any).dispose?.();
                } catch {}
              } finally {
                inFlight--;
              }
            },
            undefined,
            (err) => {
              console.error("texture load error", url, err);
              pi.status = "empty";
              inFlight--;
            }
          );
        }

        function maybeUnloadTexture(pi: PlaneItem) {
          if (pi.status !== "ready") return;
          const mat = pi.mesh.material as MeshBasicMaterial;
          if (mat.map) {
            try {
              (mat.map as any).dispose?.();
            } catch {}
            (mat as any).map = null;
            mat.color.set(0x66ccff);
            mat.needsUpdate = true;
          }
          // Reset to square placeholder
          pi.mesh.scale.set(planeSize, planeSize, 1);
          pi.texture = undefined;
          pi.status = "empty";
        }

        // Keyboard navigation (WASD + QE)
        const pressed = new Set<string>();
        const onKeyDown = (e: KeyboardEvent) => {
          pressed.add(e.key.toLowerCase());
        };
        const onKeyUp = (e: KeyboardEvent) => {
          pressed.delete(e.key.toLowerCase());
        };
        window.addEventListener("keydown", onKeyDown);
        window.addEventListener("keyup", onKeyUp);

        // Mouse look (left-drag) using quaternion orientation (no gimbal lock)
        let isDragging = false;
        let lastX = 0;
        let lastY = 0;
        let orientation = new Quaternion().copy(camera.quaternion);
        const onMouseDown = (e: MouseEvent) => {
          if (e.button !== 0) return; // left button
          isDragging = true;
          lastX = e.clientX;
          lastY = e.clientY;
        };
        const onMouseMove = (e: MouseEvent) => {
          if (!isDragging) return;
          const dx = e.clientX - lastX;
          const dy = e.clientY - lastY;
          lastX = e.clientX;
          lastY = e.clientY;
          const sensitivity = 0.0025; // radians per pixel
          const yawDelta = -dx * sensitivity; // rotate around local up
          const pitchDelta = -dy * sensitivity; // rotate around local right
          // Local axes (world-space) from current orientation
          const upAxis = new Vector3(0, 1, 0)
            .applyQuaternion(orientation)
            .normalize();
          const rightAxis = new Vector3(1, 0, 0)
            .applyQuaternion(orientation)
            .normalize();
          const qYaw = new Quaternion().setFromAxisAngle(upAxis, yawDelta);
          const qPitch = new Quaternion().setFromAxisAngle(
            rightAxis,
            pitchDelta
          );
          orientation = qYaw.multiply(qPitch).multiply(orientation);
          if (camera) camera.quaternion.copy(orientation);
        };
        const onMouseUp = () => {
          isDragging = false;
        };
        const onMouseLeave = () => {
          isDragging = false;
        };
        renderer.domElement.addEventListener("mousedown", onMouseDown);
        window.addEventListener("mousemove", onMouseMove);
        window.addEventListener("mouseup", onMouseUp);
        renderer.domElement.addEventListener("mouseleave", onMouseLeave);

        // Axes helper (tiny)
        // const axes = new AxesHelper(1); scene.add(axes);

        function onResize() {
          if (!containerRef.current || !renderer || !camera) return;
          const w = containerRef.current.clientWidth;
          const h = containerRef.current.clientHeight;
          renderer.setSize(w, h);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        }
        window.addEventListener("resize", onResize);

        // Wheel to dolly forward/backward
        const onWheel = (e: WheelEvent) => {
          if (!camera) return;
          const delta = Math.sign(e.deltaY);
          const forward = new Vector3(0, 0, -1)
            .applyQuaternion(orientation)
            .normalize();
          camera.position.addScaledVector(forward, -delta * 0.2);
        };
        renderer.domElement.addEventListener("wheel", onWheel, {
          passive: true,
        } as AddEventListenerOptions);

        let lastTime = performance.now();
        // temp vectors to avoid allocations in the loop
        const tmpWorld = new Vector3();
        const tmpNdc = new Vector3();
        const animate = (t: number) => {
          if (disposed) return;
          const dt = Math.max(0.001, Math.min(0.1, (t - lastTime) / 1000));
          lastTime = t;

          // WASD movement relative to camera orientation (quaternion-based)
          if (pressed.size > 0) {
            const forward = new Vector3(0, 0, -1)
              .applyQuaternion(orientation)
              .normalize();
            const right = new Vector3(1, 0, 0)
              .applyQuaternion(orientation)
              .normalize();

            const move = new Vector3();
            if (pressed.has("w")) move.add(forward);
            if (pressed.has("s")) move.addScaledVector(forward, -1);
            if (pressed.has("a")) move.addScaledVector(right, -1);
            if (pressed.has("d")) move.add(right);

            // Q/E roll around forward axis (apply to orientation quaternion)
            const rollSpeed = 1.2; // radians per second
            if (pressed.has("q") || pressed.has("e")) {
              const sign = pressed.has("q") ? -1 : 1;
              const qRoll = new Quaternion().setFromAxisAngle(
                forward,
                sign * rollSpeed * dt
              );
              orientation = qRoll.multiply(orientation);
              if (camera) camera.quaternion.copy(orientation);
            }

            if (move.lengthSq() > 0) {
              move.normalize();
              const base = 0.8; // base units/sec at zClip=1
              const scale = zClipRef.current;
              const speed = (pressed.has("shift") ? 3.0 : 1.0) * base * scale;
              move.multiplyScalar(speed * dt);
              if (camera) camera.position.add(move);
              // camera.quaternion already reflects orientation
            }
          }
          // Billboard planes face camera, preclip by frustum, and decide which to texture
          if (camera && renderer) {
            const h = renderer.domElement.height;
            const f =
              0.5 * h /
              Math.tan(
                MathUtils.degToRad((camera as THREE.PerspectiveCamera).fov * 0.5)
              );
            for (let i = 0; i < planes.length; i++) {
              const p = planes[i];
              // Face camera (use current orientation)
              p.mesh.quaternion.copy(orientation);
              // Preclip: skip offscreen or beyond near/far by projecting to NDC
              p.mesh.getWorldPosition(tmpWorld);
              tmpNdc.copy(tmpWorld).project(camera);
              const inFrustum =
                tmpNdc.x >= -1 && tmpNdc.x <= 1 &&
                tmpNdc.y >= -1 && tmpNdc.y <= 1 &&
                tmpNdc.z >= -1 && tmpNdc.z <= 1;
              if (!inFrustum) {
                // Hide and free texture if any
                if (p.mesh.visible) p.mesh.visible = false;
                const mat = p.mesh.material as MeshBasicMaterial;
                if ((mat as any).map) {
                  maybeUnloadTexture(p);
                }
                continue;
              } else if (!p.mesh.visible) {
                p.mesh.visible = true;
              }
              // Estimate on-screen height in pixels for current placeholder height (planeSize)
              const dist = camera.position.distanceTo(p.mesh.position);
              const screenHeightPx = (planeSize * f) / Math.max(dist, 1e-6);
              const bigEnough = screenHeightPx >= thresholdFrac * h;
              if (bigEnough) {
                maybeLoadTexture(p, i, h);
              } else {
                // If currently textured, unload to save memory
                const mat = p.mesh.material as MeshBasicMaterial;
                if ((mat as any).map) {
                  maybeUnloadTexture(p);
                }
              }
            }
          }

          if (renderer && scene && camera) {
            renderer.render(scene, camera);
          }
          animationId = requestAnimationFrame(animate);
        };
        animationId = requestAnimationFrame(animate);

        setStatus("");

        return () => {
          disposed = true;
          if (animationId !== null) cancelAnimationFrame(animationId);
          window.removeEventListener("resize", onResize);
          window.removeEventListener("keydown", onKeyDown);
          window.removeEventListener("keyup", onKeyUp);
          renderer?.domElement?.removeEventListener("wheel", onWheel);
          renderer?.domElement?.removeEventListener("mousedown", onMouseDown);
          window.removeEventListener("mousemove", onMouseMove);
          window.removeEventListener("mouseup", onMouseUp);
          renderer?.domElement?.removeEventListener("mouseleave", onMouseLeave);
          // Clean up three objects
          scene?.traverse((obj: THREE.Object3D) => {
            if ("geometry" in obj && obj.geometry) {
              (obj.geometry as THREE.BufferGeometry).dispose?.();
            }
            if ("material" in obj && obj.material) {
              const m = obj.material as THREE.Material | THREE.Material[];
              if (Array.isArray(m)) {
                m.forEach((mm) => mm.dispose?.());
              } else {
                m.dispose?.();
              }
            }
          });
          renderer?.dispose?.();
          if (renderer?.domElement?.parentNode) {
            renderer.domElement.parentNode.removeChild(renderer.domElement);
          }
        };
      } catch (e: unknown) {
        console.error(e);
        setError(e instanceof Error ? e.message : String(e));
        setStatus("");
      }
    }

    const cleanupPromise = init();
    return () => {
      // ensure async cleanup runs
      (async () => {
        const c = await cleanupPromise;
        if (typeof c === "function") c();
      })();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // React to zClip changes: adjust camera near/far planes and fog
  useEffect(() => {
    zClipRef.current = zClip;
    const cam = cameraRef.current;
    const scn = sceneRef.current;
    if (cam) {
      cam.near = Math.max(1e-6, Math.min(0.01, zClip * 0.1));
      cam.far = zClip;
      cam.updateProjectionMatrix();
    }
    if (scn && scn.fog) {
      // Update existing fog parameters
      const fog = scn.fog as THREE.Fog;
      if (fog.color?.set) fog.color.set(0x000000);
      fog.near = Math.max(0, zClip * 0.6);
      fog.far = zClip;
    }
  }, [zClip]);

  return (
    <div className="fixed inset-0 bg-black">
      <div ref={containerRef} className="absolute inset-0" />
      <div className="absolute left-4 top-4 z-10 rounded bg-black/60 px-3 py-2 text-xs text-white">
        <div className="font-semibold">3D Embedding Viewer</div>
        <div className="opacity-80">
          Drag: look around • Scroll: dolly • WASD: move • Q/E: roll • Shift:
          boost
        </div>
        <div className="mt-2 flex items-center gap-2">
          <label htmlFor="zclip" className="opacity-80">
            Far clip (log)
          </label>
          <input
            id="zclip"
            type="range"
            min={-3}
            max={1}
            step={0.01}
            value={logZ}
            onChange={(e) => {
              const v = parseFloat((e.target as HTMLInputElement).value);
              setLogZ(v);
              setZClip(Math.pow(10, v));
            }}
          />
          <span className="tabular-nums w-16 text-right">
            {zClip.toExponential(2)}
          </span>
        </div>
        {status && <div className="opacity-80 mt-1">{status}</div>}
        {error && <div className="text-red-400 mt-1">Error: {error}</div>}
      </div>
    </div>
  );
}
