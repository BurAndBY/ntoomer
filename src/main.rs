#![windows_subsystem = "windows"] //  This is needed to hide the console window

use glam::Vec2;
use glutin::{
    config::{ConfigTemplateBuilder, GlConfig},
    context::{ContextAttributesBuilder, NotCurrentGlContext},
    display::{GetGlDisplay, GlDisplay},
    surface::{GlSurface, SwapInterval, WindowSurface},
};
use glutin_winit::DisplayBuilder;
use raw_window_handle::HasRawWindowHandle;
use screenshots::Screen;
use std::{
    ffi::{c_void, CString},
    mem,
    time::Instant,
};
use winit::{
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Fullscreen, WindowBuilder},
};

// Shaders, hardcoded and copied from tsoding's boomer
const VERTEX_SHADER_SOURCE: &str = r#"
    #version 130
    in vec3 aPos;
    in vec2 aTexCoord;
    out vec2 texcoord;

    uniform vec2 cameraPos;
    uniform float cameraScale;
    uniform vec2 windowSize;
    uniform vec2 screenshotSize;

    vec3 to_world(vec3 v) {
        vec2 ratio = vec2(windowSize.x / screenshotSize.x / cameraScale, windowSize.y / screenshotSize.y / cameraScale);
        return vec3((v.x / screenshotSize.x * 2.0 - 1.0) / ratio.x, (v.y / screenshotSize.y * 2.0 - 1.0) / ratio.y, v.z);
    }

    void main() {
        gl_Position = vec4(to_world((aPos - vec3(cameraPos * vec2(1.0, -1.0), 0.0))), 1.0);
        texcoord = aTexCoord;
    }
"#;

const FRAGMENT_SHADER_SOURCE: &str = r#"
    #version 130
    out mediump vec4 color;
    in mediump vec2 texcoord;

    uniform sampler2D tex;
    uniform vec2 cursorPos;
    uniform vec2 windowSize;
    uniform float flShadow;
    uniform float flRadius;
    uniform float cameraScale;

    void main() {
        vec4 cursor = vec4(cursorPos.x, cursorPos.y, 0.0, 1.0);
        color = mix(texture(tex, texcoord), vec4(0.0, 0.0, 0.0, 0.0), length(cursor - gl_FragCoord) < (flRadius * cameraScale) ? 0.0 : flShadow);
    }
"#;

// Config, currently hardcoded
struct Config {
    min_scale: f32,
    scroll_speed: f32,
    drag_friction: f32,
    scale_friction: f32,
}

const CONFIG: Config = Config {
    min_scale: 0.01,
    scroll_speed: 1.5,
    drag_friction: 6.0,
    scale_friction: 4.0,
};

const VELOCITY_THRESHOLD: f32 = 15.0;
const INITIAL_FL_DELTA_RADIUS: f32 = 250.0;
const FL_DELTA_RADIUS_DECELERATION: f32 = 10.0;

#[derive(Default, Debug, Clone, Copy)]
struct Camera {
    position: Vec2,
    velocity: Vec2,
    scale: f32,
    delta_scale: f32,
    scale_pivot: Vec2,
}

#[derive(Debug, Clone, Copy)]
struct Flashlight {
    is_enabled: bool,
    shadow: f32,
    radius: f32,
    delta_radius: f32,
}

#[derive(Default, Debug, Clone, Copy)]
struct MouseState {
    pos: Vec2,
    is_dragging: bool,
}

struct AppState {
    camera: Camera,
    flashlight: Flashlight,
    mouse: MouseState,
    ctrl_pressed: bool,
}

impl AppState {
    fn new() -> Self {
        Self {
            camera: Camera {
                scale: 1.0,
                ..Default::default()
            },
            flashlight: Flashlight {
                is_enabled: false,
                shadow: 0.0,
                radius: 200.0,
                delta_radius: 0.0,
            },
            mouse: MouseState::default(),
            ctrl_pressed: false,
        }
    }
    fn reset(&mut self) {
        self.camera = Camera {
            scale: 1.0,
            ..Default::default()
        };
        self.flashlight.radius = 200.0;
    }
}

// OpenGL Helpers

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn compile_shader(source: &str, kind: gl::types::GLenum) -> Result<gl::types::GLuint, String> {
    let shader = gl::CreateShader(kind);
    let c_str = CString::new(source.as_bytes()).unwrap();
    gl::ShaderSource(shader, 1, &c_str.as_ptr(), std::ptr::null());
    gl::CompileShader(shader);

    let mut success = gl::FALSE as gl::types::GLint;
    gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
    if success == gl::TRUE as gl::types::GLint {
        Ok(shader)
    } else {
        let mut len = 0;
        gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
        let mut buffer = Vec::with_capacity(len as usize);
        buffer.set_len((len as usize) - 1);
        gl::GetShaderInfoLog(shader, len, std::ptr::null_mut(), buffer.as_mut_ptr() as *mut i8);
        Err(String::from_utf8(buffer).unwrap_or("Shader compile error".to_string()))
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn create_shader_program(
    vs_src: &str,
    fs_src: &str,
) -> Result<gl::types::GLuint, String> {
    let vs = compile_shader(vs_src, gl::VERTEX_SHADER)?;
    let fs = compile_shader(fs_src, gl::FRAGMENT_SHADER)?;

    let program = gl::CreateProgram();
    gl::AttachShader(program, vs);
    gl::AttachShader(program, fs);
    gl::LinkProgram(program);

    let mut success = gl::FALSE as gl::types::GLint;
    gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
    if success == gl::TRUE as gl::types::GLint {
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);
        Ok(program)
    } else {
        let mut len = 0;
        gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
        let mut buffer = Vec::with_capacity(len as usize);
        buffer.set_len((len as usize) - 1);
        gl::GetProgramInfoLog(program, len, std::ptr::null_mut(), buffer.as_mut_ptr() as *mut i8);
        Err(String::from_utf8(buffer).unwrap_or("Program link error".to_string()))
    }
}

struct UniformLocations {
    camera_pos: i32,
    camera_scale: i32,
    screenshot_size: i32,
    window_size: i32,
    cursor_pos: i32,
    fl_shadow: i32,
    fl_radius: i32,
    tex: i32,
}

fn main() {
    println!("Taking screenshot...");
    let screen = Screen::from_point(0, 0).expect("Could not find screen");
    let screenshot = screen.capture().expect("Could not take screenshot");
    let screenshot_size = Vec2::new(screenshot.width() as f32, screenshot.height() as f32);
    let screenshot_data = screenshot.rgba();
    println!("Screenshot taken: {}x{}", screenshot_size.x, screenshot_size.y);

    // Window and OpenGL context setup
    let event_loop = EventLoop::new().unwrap();
    let window_builder = WindowBuilder::new()
        .with_title("NToomer")
        .with_fullscreen(Some(Fullscreen::Borderless(None)));

    let template = ConfigTemplateBuilder::new().with_alpha_size(8);
    let display_builder = DisplayBuilder::new().with_window_builder(Some(window_builder));
    let (window, gl_config) = display_builder
        .build(&event_loop, template, |configs| {
            configs.reduce(|acc, config| {
                if config.num_samples() > acc.num_samples() { config } else { acc }
            }).unwrap()
        })
        .expect("Failed to build window");

    let window = window.unwrap();
    let raw_window_handle = window.raw_window_handle();
    let gl_display = gl_config.display();

    let context_attributes = ContextAttributesBuilder::new().build(Some(raw_window_handle));
    let not_current_gl_context = unsafe {
        gl_display.create_context(&gl_config, &context_attributes).expect("Failed to create context")
    };

    let (width, height): (u32, u32) = window.inner_size().into();
    let attrs = glutin::surface::SurfaceAttributesBuilder::<WindowSurface>::new().build(
        raw_window_handle,
        width.try_into().unwrap(),
        height.try_into().unwrap(),
    );
    let gl_surface = unsafe {
        gl_config.display().create_window_surface(&gl_config, &attrs).expect("Failed to create surface")
    };
    let gl_context = not_current_gl_context.make_current(&gl_surface).unwrap();

    gl::load_with(|symbol| {
        let c_str = CString::new(symbol).unwrap();
        gl_display.get_proc_address(&c_str)
    });
    gl_surface.set_swap_interval(&gl_context, SwapInterval::Wait(1.try_into().unwrap())).unwrap();

    // Initialize app
    let mut app_state = AppState::new();
    let mut last_frame = Instant::now();

    // I haven't figured out a better way to keep the shaders safe, so this will do
    unsafe {
        let shader_program = create_shader_program(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
            .expect("Shader program creation failed");

        let get_loc = |name: &str| {
            let c_name = CString::new(name).unwrap();
            gl::GetUniformLocation(shader_program, c_name.as_ptr())
        };

        let uniforms = UniformLocations {
            camera_pos: get_loc("cameraPos"),
            camera_scale: get_loc("cameraScale"),
            screenshot_size: get_loc("screenshotSize"),
            window_size: get_loc("windowSize"),
            cursor_pos: get_loc("cursorPos"),
            fl_shadow: get_loc("flShadow"),
            fl_radius: get_loc("flRadius"),
            tex: get_loc("tex"),
        };

        // Setup geometry
        let (w, h) = (screenshot_size.x, screenshot_size.y);
        let verts: [f32; 20] = [
            // pos        // tex
            w, 0.0, 0.0, 1.0, 1.0, // Top Right
            w, h, 0.0, 1.0, 0.0, // Bottom Right
            0.0, h, 0.0, 0.0, 0.0, // Bottom Left
            0.0, 0.0, 0.0, 0.0, 1.0, // Top Left
        ];
        let inds: [u32; 6] = [0, 1, 3, 1, 2, 3];

        let (mut vao, mut vbo, mut ebo) = (0, 0, 0);
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);
        gl::GenBuffers(1, &mut ebo);

        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (verts.len() * mem::size_of::<f32>()) as isize,
            verts.as_ptr() as *const c_void,
            gl::STATIC_DRAW,
        );

        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
        gl::BufferData(
            gl::ELEMENT_ARRAY_BUFFER,
            (inds.len() * mem::size_of::<u32>()) as isize,
            inds.as_ptr() as *const c_void,
            gl::STATIC_DRAW,
        );

        let stride = (5 * mem::size_of::<f32>()) as i32;
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, stride, std::ptr::null());
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, stride, (3 * mem::size_of::<f32>()) as *const c_void);
        gl::EnableVertexAttribArray(1);

        let mut texture = 0;
        gl::GenTextures(1, &mut texture);
        gl::BindTexture(gl::TEXTURE_2D, texture);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_BORDER as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_BORDER as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);

        gl::TexImage2D(
            gl::TEXTURE_2D, 0, gl::RGBA as i32, w as i32, h as i32, 0,
            gl::RGBA, gl::UNSIGNED_BYTE, screenshot_data.as_ptr() as *const c_void,
        );

        // Main event loop
        let _ = event_loop.run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::KeyboardInput { event: KeyEvent { logical_key, state: ElementState::Pressed, .. }, .. } => {
                        match logical_key.as_ref() {
                            // q = quit
                            // 0 - reset
                            // f - flashlight

                            Key::Named(NamedKey::Escape) | Key::Character("q") => elwt.exit(),
                            Key::Named(NamedKey::Control) => app_state.ctrl_pressed = true,
                            Key::Character("0") => app_state.reset(),
                            Key::Character("f") => app_state.flashlight.is_enabled = !app_state.flashlight.is_enabled,
                            _ => {}
                        }
                    }
                    WindowEvent::KeyboardInput { event: KeyEvent { logical_key, state: ElementState::Released, .. }, .. } => {
                        if logical_key == Key::Named(NamedKey::Control) {
                            app_state.ctrl_pressed = false;
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        // Convert winit's top-left origin to OpenGL's bottom-left
                        let window_size = window.inner_size();
                        app_state.mouse.pos = Vec2::new(position.x as f32, window_size.height as f32 - position.y as f32);
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        if button == MouseButton::Left {
                            match state {
                                ElementState::Pressed => {
                                    app_state.mouse.is_dragging = true;
                                    app_state.camera.velocity = Vec2::ZERO;
                                }
                                ElementState::Released => {
                                    app_state.mouse.is_dragging = false;
                                }
                            }
                        }
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let scroll_y = if let MouseScrollDelta::LineDelta(_, y) = delta { y } else { 0.0 };
                        if scroll_y == 0.0 { return; }

                        if app_state.ctrl_pressed && app_state.flashlight.is_enabled {
                            app_state.flashlight.delta_radius += scroll_y * INITIAL_FL_DELTA_RADIUS;
                        } else {
                            let window_size = window.inner_size();
                            app_state.camera.scale_pivot = Vec2::new(app_state.mouse.pos.x, window_size.height as f32 - app_state.mouse.pos.y);
                            app_state.camera.delta_scale += scroll_y * CONFIG.scroll_speed;
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        // Window update
                        let dt = last_frame.elapsed().as_secs_f32();
                        last_frame = Instant::now();
                        let cam = &mut app_state.camera;
                        let fl = &mut app_state.flashlight;

                        if cam.delta_scale.abs() > 0.01 {
                            let window_size = Vec2::new(window.inner_size().width as f32, window.inner_size().height as f32);
                            let p0 = (cam.scale_pivot - window_size * 0.5) / cam.scale;
                            let new_scale = cam.scale + cam.delta_scale * dt;
                            cam.scale = new_scale.max(CONFIG.min_scale);
                            let p1 = (cam.scale_pivot - window_size * 0.5) / cam.scale;
                            cam.position += p0 - p1;
                            cam.delta_scale -= cam.delta_scale * dt * CONFIG.scale_friction;
                        }
                        if !app_state.mouse.is_dragging && cam.velocity.length() > VELOCITY_THRESHOLD {
                            cam.position += cam.velocity * dt;
                            cam.velocity -= cam.velocity * dt * CONFIG.drag_friction;
                        }
                        if fl.delta_radius.abs() > 1.0 {
                            fl.radius = (fl.radius + fl.delta_radius * dt).max(0.0);
                            fl.delta_radius -= fl.delta_radius * dt * FL_DELTA_RADIUS_DECELERATION;
                        }
                        if fl.is_enabled {
                            fl.shadow = (fl.shadow + 6.0 * dt).min(0.8);
                        } else {
                            fl.shadow = (fl.shadow - 6.0 * dt).max(0.0);
                        }

                        let window_size = window.inner_size();
                        gl::Viewport(0, 0, window_size.width as i32, window_size.height as i32);
                        gl::ClearColor(0.0, 0.0, 0.0, 1.0);
                        gl::Clear(gl::COLOR_BUFFER_BIT);

                        gl::UseProgram(shader_program);
                        gl::Uniform2f(uniforms.camera_pos, cam.position.x, cam.position.y);
                        gl::Uniform1f(uniforms.camera_scale, cam.scale);
                        gl::Uniform2f(uniforms.screenshot_size, screenshot_size.x, screenshot_size.y);
                        gl::Uniform2f(uniforms.window_size, window_size.width as f32, window_size.height as f32);
                        gl::Uniform2f(uniforms.cursor_pos, app_state.mouse.pos.x, app_state.mouse.pos.y);
                        gl::Uniform1f(uniforms.fl_shadow, fl.shadow);
                        gl::Uniform1f(uniforms.fl_radius, fl.radius);

                        gl::ActiveTexture(gl::TEXTURE0);
                        gl::BindTexture(gl::TEXTURE_2D, texture);
                        gl::Uniform1i(uniforms.tex, 0);

                        gl::BindVertexArray(vao);
                        gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());

                        gl_surface.swap_buffers(&gl_context).unwrap();
                    }
                    _ => (),
                },
                Event::AboutToWait => {
                    window.request_redraw();
                }
                Event::DeviceEvent { event: winit::event::DeviceEvent::MouseMotion { delta }, .. } => {
                    if app_state.mouse.is_dragging {

                        // Replace the delta.NO with -delta.NO to flip axis (0 = x, 1 = y)
                        let delta_world = Vec2::new(delta.0 as f32, delta.1 as f32) / app_state.camera.scale;
                        app_state.camera.position -= delta_world;
                        app_state.camera.velocity = (delta_world / (1.0/120.0)) * -1.0;
                    }
                }
                _ => (),
            }
        });
    }
}