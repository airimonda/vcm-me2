"use strict";
/* Watson dashboard (VCM-ME2) — Apple HIG plain style. Vanilla JS, no frameworks, no CDN.
 * Talks to runtime/ui_server.py over /ws (see that module for the message contract).
 * The conversation log shows "Understood as": a sentence the runtime rebuilds from
 * command + slot. It is never a transcript (the model does not transcribe).
 * URL options: ?theme=dark|light, ?test=1 (test panel in Diagnostics), ?lowfx=1,
 * ?mascot=<idle|listening|thinking|speaking|happy|oops> (pins the orb/glow for screenshots).
 */

(function () {
  var params = new URLSearchParams(location.search);
  var TEST_MODE = params.get("test") === "1";

  var els = {
    reconnectPill: document.getElementById("reconnect-pill"),

    sidebarToggle: document.getElementById("sidebar-toggle"),
    toolbarTitle: document.getElementById("toolbar-title"),
    clearActivity: document.getElementById("clear-activity"),
    perfToggle: document.getElementById("perf-toggle"),
    perfPopover: document.getElementById("perf-popover"),
    perfReadout: document.getElementById("perf-readout"),

    musicDot: document.getElementById("music-dot"),
    musicDotText: document.getElementById("music-dot-text"),
    musicTrack: document.getElementById("music-track"),
    musicVolume: document.getElementById("music-volume"),
    musicMeta: document.getElementById("music-meta"),
    accMusic: document.getElementById("acc-music"),
    accMusicState: document.getElementById("acc-music-state"),

    lightState: document.getElementById("light-state"),
    lightMeta: document.getElementById("light-meta"),
    lightIconCircle: document.getElementById("light-icon-circle"),
    accLight: document.getElementById("acc-light"),
    accLightState: document.getElementById("acc-light-state"),
    accLightIconCircle: document.getElementById("acc-light-icon-circle"),

    thermoSetpoint: document.getElementById("thermo-setpoint"),
    thermoRoom: document.getElementById("thermo-room"),
    accThermo: document.getElementById("acc-thermo"),
    accThermoState: document.getElementById("acc-thermo-state"),

    weatherTemp: document.getElementById("weather-temp"),
    weatherMeta: document.getElementById("weather-meta"),
    clockNow: document.getElementById("clock-now"),
    accWeather: document.getElementById("acc-weather"),
    accWeatherState: document.getElementById("acc-weather-state"),
    accWeatherCaption: document.getElementById("acc-weather-caption"),

    phoneTile: document.getElementById("home-phone"),
    phoneInfo: document.getElementById("phone-info"),
    accPhone: document.getElementById("acc-phone"),
    accPhoneState: document.getElementById("acc-phone-state"),

    nowTimersList: document.getElementById("now-timers-list"),
    nowRemindersList: document.getElementById("now-reminders-list"),
    timersOnlyList: document.getElementById("timers-only-list"),
    alarmsOnlyList: document.getElementById("alarms-only-list"),
    remindersOnlyList: document.getElementById("reminders-only-list"),

    exchangeLastAnswer: document.getElementById("exchange-last-answer"),

    perfE2e: document.getElementById("perf-e2e"),
    perfVcmWake: document.getElementById("perf-vcm-wake"),
    perfSys: document.getElementById("perf-sys"),
    perfStages: document.getElementById("perf-stages"),
    perfAccuracy: document.getElementById("perf-accuracy"),
    perfWarning: document.getElementById("perf-warning"),
    perfCard: document.getElementById("perf-card"),
    sparkE2e: document.querySelector("#spark-e2e polyline"),

    popE2e: document.getElementById("pop-e2e"),
    popVcm: document.getElementById("pop-vcm"),
    popWake: document.getElementById("pop-wake"),
    popCpu: document.getElementById("pop-cpu"),
    popRam: document.getElementById("pop-ram"),
    popTemp: document.getElementById("pop-temp"),
    popTurns: document.getElementById("pop-turns"),
    popAccuracy: document.getElementById("pop-accuracy"),
    popFalseWakes: document.getElementById("pop-falsewakes"),

    diagTurns: document.getElementById("diag-turns"),
    diagAccuracy: document.getElementById("diag-accuracy"),
    diagFalseWakes: document.getElementById("diag-falsewakes"),
    diagRejection: document.getElementById("diag-rejection"),
    diagCpuLast: document.getElementById("diag-cpu-last"),
    diagRamLast: document.getElementById("diag-ram-last"),
    diagTempLast: document.getElementById("diag-temp-last"),

    logList: document.getElementById("log-list"),

    testPanel: document.getElementById("test-panel"),
    testGroups: document.getElementById("test-groups"),

    activityTbody: document.getElementById("activity-tbody"),
    activityEmpty: document.getElementById("activity-empty"),
    nowActivityList: document.getElementById("now-activity-list"),

    exchangeEmpty: document.getElementById("exchange-empty"),
    exchangeBody: document.getElementById("exchange-body"),
    exchangeHeardText: document.getElementById("exchange-heard-text"),
    exchangeHeardMeta: document.getElementById("exchange-heard-meta"),
    exchangeReplyText: document.getElementById("exchange-reply-text"),
    exchangePlayHeard: document.getElementById("exchange-play-heard"),
    exchangePlayReply: document.getElementById("exchange-play-reply"),
    exchangeMarkYes: document.getElementById("exchange-mark-yes"),
    exchangeMarkNo: document.getElementById("exchange-mark-no"),
    exchangePicker: document.getElementById("exchange-picker")
  };

  var ORBS = document.querySelectorAll(".js-orb");
  var MODE_LABELS_EL = document.querySelectorAll(".js-mode-label");

  var MODE_LABELS = {
    idle: "Idle", listening: "Listening", thinking: "Thinking",
    speaking: "Speaking", happy: "Done", oops: "Didn't catch that"
  };
  var GLOW_STATE = { idle: null, listening: "listening", thinking: "thinking", speaking: "speaking", happy: "done", oops: "oops" };
  var MASCOT_MODES = ["idle", "listening", "thinking", "speaking", "happy", "oops"];
  var MASCOT_MIN_DWELL = 700; // ms a pose must stay up before the next queued one can replace it
  var FORCE_MASCOT = params.get("mascot");
  if (MASCOT_MODES.indexOf(FORCE_MASCOT) === -1) FORCE_MASCOT = null;

  /* ---------------- view routing ---------------- */

  var VIEWS = ["now", "home", "timers", "activity", "diagnostics"];
  var VIEW_TITLES = { now: "Now", home: "Home", timers: "Timers", activity: "Activity", diagnostics: "Diagnostics" };
  var sidebarLinks = document.querySelectorAll(".sidebar-item");
  var tabbarLinks = document.querySelectorAll(".tabbar-item");
  var viewSections = document.querySelectorAll(".view");

  function currentView() {
    var h = location.hash.replace("#", "");
    return VIEWS.indexOf(h) !== -1 ? h : "now";
  }

  function applyRoute() {
    var view = currentView();
    viewSections.forEach(function (sec) { sec.hidden = sec.dataset.view !== view; });
    [sidebarLinks, tabbarLinks].forEach(function (list) {
      list.forEach(function (a) {
        if (a.dataset.view === view) a.setAttribute("aria-current", "page");
        else a.removeAttribute("aria-current");
      });
    });
    els.toolbarTitle.textContent = VIEW_TITLES[view] || "Watson";
    els.clearActivity.hidden = view !== "activity";
    if (els.perfPopover && !els.perfPopover.hidden) closePerfPopover();
  }
  window.addEventListener("hashchange", applyRoute);
  applyRoute();

  /* ---------------- sidebar toggle ---------------- */

  var appShell = document.querySelector(".app-shell");
  if (els.sidebarToggle) {
    els.sidebarToggle.addEventListener("click", function () {
      var hidden = appShell.classList.toggle("sidebar-hidden");
      els.sidebarToggle.setAttribute("aria-expanded", hidden ? "false" : "true");
    });
  }

  /* ---------------- perf popover ---------------- */

  function closePerfPopover() {
    els.perfPopover.hidden = true;
    els.perfToggle.setAttribute("aria-expanded", "false");
  }
  if (els.perfToggle) {
    els.perfToggle.addEventListener("click", function () {
      var willOpen = els.perfPopover.hidden;
      els.perfPopover.hidden = !willOpen;
      els.perfToggle.setAttribute("aria-expanded", willOpen ? "true" : "false");
    });
    document.addEventListener("click", function (ev) {
      if (!els.perfPopover.hidden && !ev.target.closest(".perf-popover-wrap")) closePerfPopover();
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") closePerfPopover();
    });
  }

  /* ---------------- test panel ---------------- */

  var SLOTS = {
    TIMER: ["10 seconds", "30 seconds", "1 minute"],
    ALARM: ["6:00 AM", "8:00 AM", "9:00 PM"],
    TEMPERATURE: ["18 degrees", "22 degrees", "26 degrees"],
    BRIGHTNESS: ["20 percent", "60 percent", "100 percent"],
    COLOR: ["Red", "Blue", "Green"],
    CREATE_REMINDER: ["Drink water", "Study", "Exercise"]
  };
  var COMMANDS = ["PLAY_MUSIC", "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN", "LIGHT_ON", "LIGHT_OFF",
    "WEATHER", "TIME", "CALL", "MESSAGE", "LIST_REMINDERS", "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS",
    "COLOR", "CREATE_REMINDER"];
  var COMMAND_LIST = COMMANDS.concat(["OUT_OF_SCOPE"]);
  var TEST_GROUPS = [
    { title: "Music", commands: ["PLAY_MUSIC", "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN"] },
    { title: "Info", commands: ["WEATHER", "TIME"] },
    { title: "Light", commands: ["LIGHT_ON", "LIGHT_OFF", "BRIGHTNESS", "COLOR"] },
    { title: "Thermostat", commands: ["TEMPERATURE"] },
    { title: "Clock", commands: ["TIMER", "ALARM", "CREATE_REMINDER", "LIST_REMINDERS"] },
    { title: "Phone", commands: ["CALL", "MESSAGE"] },
    { title: "Other", commands: ["OUT_OF_SCOPE"] }
  ];
  var TEST_INTENTS = TEST_GROUPS.map(function (g) {
    var buttons = [];
    g.commands.forEach(function (c) {
      if (SLOTS[c]) SLOTS[c].forEach(function (v) { buttons.push({ label: c.replace(/_/g, " ").toLowerCase() + " · " + v, command: c, slot: v }); });
      else buttons.push({ label: c.replace(/_/g, " ").toLowerCase(), command: c, slot: null });
    });
    return { intent: g.title, buttons: buttons };
  });

  function buildTestPanel() {
    if (!TEST_MODE) return;
    els.testPanel.hidden = false;
    TEST_INTENTS.forEach(function (group) {
      var wrap = document.createElement("div");
      wrap.className = "test-group";
      var h4 = document.createElement("h4");
      h4.textContent = group.intent;
      wrap.appendChild(h4);
      group.buttons.forEach(function (b) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "pill";
        btn.textContent = b.label;
        btn.addEventListener("click", function () {
          send({ type: "command", command: b.command, slot: b.slot });
        });
        wrap.appendChild(btn);
      });
      els.testGroups.appendChild(wrap);
    });
  }
  buildTestPanel();

  /* ---------------- websocket ---------------- */

  var ws = null;
  var reconnectDelay = 1000;

  function wsUrl() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    return proto + "//" + location.host + "/ws";
  }

  function connect() {
    ws = new WebSocket(wsUrl());
    ws.addEventListener("open", function () {
      reconnectDelay = 1000;
      els.reconnectPill.hidden = true;
    });
    ws.addEventListener("message", function (ev) {
      var msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      handleMessage(msg);
    });
    ws.addEventListener("close", scheduleReconnect);
    ws.addEventListener("error", function () { try { ws.close(); } catch (e) {} });
  }

  function scheduleReconnect() {
    els.reconnectPill.hidden = false;
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 1.6, 8000);
  }

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(obj));
    }
  }

  connect();

  /* ---------------- message dispatch ---------------- */

  function handleMessage(msg) {
    switch (msg.type) {
      case "state": renderState(msg.state || {}); break;
      case "heard": renderHeard(msg); break;
      case "reply": renderReplyLog(msg); break;
      case "turn": appendTurn(msg); break;
      case "answer": renderAnswer(msg); break;
      case "metrics": renderMetrics(msg); break;
      case "log": appendLog(msg.level, msg.msg); break;
      default: break;
    }
  }

  /* ---------------- mode / orb / window glow ---------------- */

  var currentMode = "idle";
  var flashTimer = null;
  var glowFlashTimer = null;

  var mascotShownAt = 0;
  var mascotQueuedState = null;
  var mascotQueuedLabel = null;
  var mascotQueueTimer = null;

  function paintOrbs(state) {
    ORBS.forEach(function (el) { el.setAttribute("data-state", state); });
  }
  function paintLabels(text) {
    MODE_LABELS_EL.forEach(function (el) { el.textContent = text; });
  }
  function paintGlow(state) {
    var glow = GLOW_STATE[state];
    if (state === "happy" || state === "oops") {
      clearTimeout(glowFlashTimer);
      document.body.setAttribute("data-glow", glow);
      glowFlashTimer = setTimeout(function () {
        document.body.setAttribute("data-glow", GLOW_STATE[currentMode] || "");
      }, 900);
    } else {
      document.body.setAttribute("data-glow", glow || "");
    }
  }

  function paintMascotState(state, label) {
    paintOrbs(state);
    paintLabels(label);
    paintGlow(state);
    mascotShownAt = Date.now();
  }

  function showMascotState(state, label) {
    var elapsed = Date.now() - mascotShownAt;
    if (elapsed >= MASCOT_MIN_DWELL) {
      if (mascotQueueTimer) { clearTimeout(mascotQueueTimer); mascotQueueTimer = null; }
      paintMascotState(state, label);
      return;
    }
    mascotQueuedState = state;
    mascotQueuedLabel = label;
    if (!mascotQueueTimer) {
      mascotQueueTimer = setTimeout(function () {
        mascotQueueTimer = null;
        var s = mascotQueuedState, l = mascotQueuedLabel;
        mascotQueuedState = null;
        mascotQueuedLabel = null;
        if (s !== null) paintMascotState(s, l);
      }, MASCOT_MIN_DWELL - elapsed);
    }
  }

  function setMode(mode) {
    currentMode = mode;
    if (FORCE_MASCOT) return; // test override pins the mascot; ignore live state
    if (!flashTimer) showMascotState(mode, MODE_LABELS[mode] || mode);
  }

  // happy/oops are one-off flashes: a beat of their own pose/glow, then back
  // to whatever mode was live when the flash ends.
  function flashMascot(state) {
    if (FORCE_MASCOT) return; // test override pins the mascot; ignore live events
    showMascotState(state, MODE_LABELS[state] || state);
    clearTimeout(flashTimer);
    flashTimer = setTimeout(function () {
      flashTimer = null;
      showMascotState(currentMode, MODE_LABELS[currentMode] || currentMode);
    }, 750);
  }

  // ?mascot=<state> forces a single pose for screenshotting/visual QA,
  // bypassing live websocket state entirely. Kept intentionally.
  if (FORCE_MASCOT) {
    paintOrbs(FORCE_MASCOT);
    paintLabels(MODE_LABELS[FORCE_MASCOT] || FORCE_MASCOT);
    document.body.setAttribute("data-glow", GLOW_STATE[FORCE_MASCOT] || "");
  }

  function renderHeard(msg) {
    flashMascot(msg.accepted ? "happy" : "oops");
  }

  function renderReplyLog(msg) {
    appendLog("info", "Said: " + (msg.say || ""));
  }

  /* ---------------- state -> tiles ---------------- */

  var lastMusicConnected = null;

  function renderState(state) {
    if (state.mode) setMode(state.mode);
    renderMusic(state.music || {});
    renderLight(state.light || {});
    renderThermostat(state.thermostat || {});
    renderTimersAlarms(state.timers || [], state.alarms || [], state.ringing || null);
    renderReminders(state.reminders || []);
    renderPhone(state.call || null);
    renderWeather(state.weather || null);
  }

  var MUSIC_ERRORS = {
    offline: "Can't reach Spotify", auth: "Spotify needs setup", not_configured: "Spotify not set up",
    no_device: "Speaker not found on Spotify", premium: "Spotify Premium required", api: "Spotify error", server: "Spotify not responding"
  };

  function renderMusic(m) {
    els.musicDot.classList.toggle("is-on", !!m.connected);
    els.musicDotText.textContent = m.connected ? "Connected" : "Not connected";
    if (m.connected === false && lastMusicConnected) appendLog("warn", "Spotify disconnected");
    lastMusicConnected = !!m.connected;

    var track = m.track || null;
    var label;
    if (m.error && MUSIC_ERRORS[m.error] && !m.playing) label = MUSIC_ERRORS[m.error];
    else if (track && track.title && (m.playing || m.paused)) label = track.title + (track.artist ? " — " + track.artist : "");
    else if (m.stopped) label = "Stopped";
    else if (m.playing) label = "Playing";
    else label = "Nothing playing";
    els.musicTrack.textContent = label;
    els.accMusicState.textContent = (m.paused && !m.error ? "Paused · " : "") + (track && track.title && (m.playing || m.paused) ? track.title : label);

    var vol = typeof m.volume === "number" ? m.volume : null;
    els.musicVolume.style.width = (vol == null ? 0 : vol) + "%";
    els.musicMeta.textContent = "volume " + (vol == null ? "—" : vol + "%") + (m.device ? " · " + m.device : "");

    els.accMusic.classList.toggle("is-on", !!m.playing);
  }

  function renderLight(lt) {
    els.accLight.classList.toggle("is-on", !!lt.on);
    els.lightState.textContent = lt.on ? "On" : "Off";
    var brightness = typeof lt.brightness === "number" ? lt.brightness : 100;
    var color = lt.color || "#FFE38A";
    if (lt.on) {
      els.lightIconCircle.style.setProperty("--bulb-color", color);
      els.accLightIconCircle.style.setProperty("--bulb-color", color);
      els.accLight.style.setProperty("--bulb-color", color);
      els.lightMeta.textContent = (lt.color_name || lt.color || "") + " · " + brightness + "%";
      els.accLightState.textContent = "On · " + brightness + "%";
    } else {
      els.lightIconCircle.style.removeProperty("--bulb-color");
      els.accLightIconCircle.style.removeProperty("--bulb-color");
      els.accLight.style.removeProperty("--bulb-color");
      els.lightMeta.textContent = "Off · last " + (lt.color_name || "white") + " " + brightness + "%";
      els.accLightState.textContent = "Off";
    }
  }

  function renderThermostat(ac) {
    var setpointText = (typeof ac.setpoint === "number" ? ac.setpoint : "—") + "°";
    els.thermoSetpoint.textContent = setpointText;
    var room = typeof ac.room_temp === "number" ? ac.room_temp.toFixed(1) + "°" : "—";
    els.thermoRoom.textContent = "Room " + room + " · " + (ac.on ? "On" : "Off");
    els.accThermo.classList.toggle("is-on", !!ac.on);
    els.accThermoState.textContent = ac.on ? setpointText + " · room " + room : "Off";
  }

  function renderPhone(call) {
    var isMessage = !!call && call.action === "message";
    els.phoneTile.classList.toggle("is-call", !!call && !isMessage);
    els.phoneTile.classList.toggle("is-message", isMessage);
    els.accPhone.classList.toggle("is-call", !!call && !isMessage);
    els.accPhone.classList.toggle("is-message", isMessage);
    els.accPhone.classList.toggle("is-on", !!call);
    var text;
    if (!call) text = "No active call";
    else if (call.action === "call") text = "Calling " + (call.contact || "…");
    else text = "Message to " + (call.contact || "") + (call.text ? ": " + call.text : "");
    els.phoneInfo.textContent = text;
    els.accPhoneState.textContent = text;
  }

  function renderWeather(w) {
    if (!w) return;
    els.weatherTemp.textContent = (typeof w.temp === "number" ? w.temp : "—") + "°";
    els.weatherMeta.textContent = (w.condition || "") + (w.city ? " · " + w.city : "");
    els.accWeatherState.textContent = (typeof w.temp === "number" ? w.temp + "° " : "") + (w.condition || "");
    els.accWeather.classList.add("is-on");
  }

  function tickClock() {
    if (els.clockNow) els.clockNow.textContent = "Device time " + new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  tickClock();
  setInterval(tickClock, 15000);

  /* ---------------- timers / alarms / reminders ---------------- */

  function remainSecFromEndsAt(endsAt) {
    var endsAtMs = endsAt * (endsAt < 1e12 ? 1000 : 1);
    return Math.max(0, Math.round((endsAtMs - Date.now()) / 1000));
  }
  function fmtCountdownSec(remain) {
    var m = Math.floor(remain / 60);
    var s = remain % 60;
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  var liveTimers = [];
  var liveAlarms = [];
  var lastRinging = null;

  function renderTimersAlarms(timers, alarms, ringing) {
    liveTimers = timers || [];
    liveAlarms = alarms || [];
    lastRinging = ringing;
    drawAllTimerViews();
  }

  function timerItem(t) {
    var isRinging = !!(lastRinging && lastRinging.kind === "timer" && lastRinging.id === t.id);
    var remain = t.ends_at ? remainSecFromEndsAt(t.ends_at) : null;
    return {
      label: t.label || "Timer",
      time: remain != null ? fmtCountdownSec(remain) : "",
      done: remain === 0 && !isRinging,
      ringing: isRinging
    };
  }
  function alarmItem(a) {
    return {
      label: "Alarm",
      time: ((a.time || "") + " " + (a.ampm || "").toUpperCase()).trim(),
      done: false,
      ringing: !!(lastRinging && lastRinging.kind === "alarm" && lastRinging.id === a.id)
    };
  }

  function renderEntryRows(ul, items, emptyText) {
    ul.innerHTML = "";
    if (!items.length) {
      var li = document.createElement("li");
      li.className = "entry-empty";
      li.textContent = emptyText;
      ul.appendChild(li);
      return;
    }
    items.forEach(function (it) {
      var row = document.createElement("li");
      row.className = "entry-row" + (it.ringing ? " is-ringing" : "");
      var label = document.createElement("span");
      label.className = "entry-label";
      label.textContent = it.label;
      row.appendChild(label);
      if (it.ringing) {
        var wrap = document.createElement("span");
        wrap.className = "entry-ringing";
        var ringingLabel = document.createElement("span");
        ringingLabel.className = "entry-time entry-time--ringing";
        ringingLabel.textContent = "Ringing";
        wrap.appendChild(ringingLabel);
        var stopBtn = document.createElement("button");
        stopBtn.type = "button";
        stopBtn.className = "pill--stop";
        stopBtn.textContent = "Stop";
        stopBtn.setAttribute("aria-label", "Stop " + it.label);
        stopBtn.addEventListener("click", function () { send({ type: "dismiss" }); });
        wrap.appendChild(stopBtn);
        row.appendChild(wrap);
      } else {
        var time = document.createElement("span");
        time.className = "entry-time" + (it.done ? " entry-time--done" : "");
        time.textContent = it.done ? "Done" : it.time;
        row.appendChild(time);
      }
      ul.appendChild(row);
    });
  }

  function drawAllTimerViews() {
    var timerItems = liveTimers.map(timerItem);
    var alarmItems = liveAlarms.map(alarmItem);
    renderEntryRows(els.nowTimersList, timerItems.concat(alarmItems), "No timers or alarms");
    renderEntryRows(els.timersOnlyList, timerItems, "No timers");
    renderEntryRows(els.alarmsOnlyList, alarmItems, "No alarms");
  }
  setInterval(drawAllTimerViews, 1000);

  function renderReminders(list) {
    list = list || [];
    var items = list.slice(0, 5).map(function (r) {
      return { label: r.topic || "Reminder", time: r.created ? new Date(r.created * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "", done: false, ringing: false };
    });
    renderEntryRows(els.nowRemindersList, items, "You have no reminders");
    renderEntryRows(els.remindersOnlyList, items, "You have no reminders");
  }

  /* ---------------- answer (Last answer line, kept per §4) ---------------- */

  function renderAnswer(msg) {
    if (!msg.text) { els.exchangeLastAnswer.hidden = true; return; }
    els.exchangeLastAnswer.textContent = "Last answer: " + msg.text;
    els.exchangeLastAnswer.hidden = false;
    if (msg.kind === "weather") renderWeather({ temp: msg.temp, condition: msg.condition, city: msg.city });
  }

  /* ---------------- metrics + diagnostics history ---------------- */

  var HIST_MAX = 150;
  var hist = { e2e: [], vcm: [], wake: [], cpu: [], ram: [], temp: [] };
  var SPARK_MAX = 24;
  var sparkE2eHistory = [];
  var turnsCount = 0;
  var ignoredCount = 0;

  function pushHist(key, val) {
    if (typeof val !== "number") return;
    var a = hist[key];
    a.push(val);
    if (a.length > HIST_MAX) a.shift();
  }

  function percentile(arr, p) {
    if (!arr.length) return null;
    var sorted = arr.slice().sort(function (a, b) { return a - b; });
    var idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
    return sorted[idx];
  }

  function fmtMs(v) { return typeof v === "number" ? Math.round(v) + " ms" : "—"; }

  function sparkPoints(arr) {
    if (!arr.length) return "";
    var max = Math.max.apply(null, arr) || 1;
    var min = Math.min.apply(null, arr);
    var range = Math.max(max - min, 1);
    var step = 100 / Math.max(arr.length - 1, 1);
    return arr.map(function (v, i) {
      var x = (i * step).toFixed(1);
      var y = (max === min ? 14 : 26 - ((v - min) / range) * 24).toFixed(1);
      return x + "," + y;
    }).join(" ");
  }

  var STAGE_ROWS = [
    ["wake_infer", "Wake model"], ["prep", "Window prep"], ["vcm_infer", "Command model"],
    ["decode_dispatch", "Decide"], ["actuator", "Actuator"], ["reply_start", "Reply start"],
    ["end_to_end", "End-to-end"], ["speech_end_to_reply", "Speech end to reply"]
  ];

  function renderStages(stages) {
    if (!els.perfStages) return;
    els.perfStages.innerHTML = "";
    STAGE_ROWS.forEach(function (row) {
      var st = stages[row[0]] || {};
      if (typeof st.last !== "number" && typeof st.p50 !== "number") return;
      var wrap = document.createElement("div");
      var dt = document.createElement("dt");
      dt.textContent = row[1];
      var dd = document.createElement("dd");
      dd.textContent = fmtMs(st.last) + (typeof st.p50 === "number" ? "  (p50 " + Math.round(st.p50) + ")" : "");
      wrap.appendChild(dt);
      wrap.appendChild(dd);
      els.perfStages.appendChild(wrap);
    });
  }

  function renderMetrics(msg) {
    var e2e = msg.end_to_end || {};
    var vcm = msg.vcm_infer || {};     // command model
    var wake = msg.wake_infer || {};

    if (typeof e2e.last === "number") { pushHist("e2e", e2e.last); sparkE2eHistory.push(e2e.last); if (sparkE2eHistory.length > SPARK_MAX) sparkE2eHistory.shift(); }
    if (typeof vcm.last === "number") pushHist("vcm", vcm.last);
    if (typeof wake.last === "number") pushHist("wake", wake.last);
    if (typeof msg.cpu_pct === "number") pushHist("cpu", msg.cpu_pct);
    if (typeof msg.ram_mb === "number") pushHist("ram", msg.ram_mb);
    if (typeof msg.soc_temp_c === "number") pushHist("temp", msg.soc_temp_c);

    var cpuText = typeof msg.cpu_pct === "number" ? msg.cpu_pct.toFixed(0) + "%" : "—";
    var ramText = typeof msg.ram_mb === "number" ? Math.round(msg.ram_mb) + " MB" : "—";
    var tempText = typeof msg.soc_temp_c === "number" ? msg.soc_temp_c.toFixed(0) + " °C" : "—";
    var accuracyText = typeof msg.marked_accuracy === "number" ? Math.round(msg.marked_accuracy * 100) + "%" : "—";
    var turns = msg.turns != null ? msg.turns : 0;
    var falseWakes = msg.false_wakes != null ? msg.false_wakes : 0;

    // Now: performance card
    els.perfE2e.textContent = fmtMs(e2e.last);
    els.sparkE2e.setAttribute("points", sparkPoints(sparkE2eHistory));
    els.perfVcmWake.textContent = fmtMs(vcm.last) + " command model · " + fmtMs(wake.last) + " wake";
    renderStages(msg.stages || {});
    els.perfSys.textContent = "CPU " + cpuText + " · " + ramText + " · " + tempText;
    els.perfAccuracy.textContent = "Accuracy " + accuracyText + " · " + falseWakes + " false wakes";

    var e2ep50 = typeof e2e.p50 === "number" ? e2e.p50 : percentile(hist.e2e, 0.5);
    var isThrottled = !!msg.throttled;
    var isWarn = (typeof e2ep50 === "number" && e2ep50 > 1500) || isThrottled;
    els.perfCard.classList.toggle("is-warn", isWarn);
    els.perfWarning.hidden = !isWarn;
    if (isWarn) {
      els.perfWarning.textContent = isThrottled
        ? "The Pi is reporting thermal throttling."
        : "End-to-end latency is running above 1.5 s.";
    }

    // toolbar: compact readout + full popover
    els.perfReadout.textContent = fmtMs(e2e.last) + " · " + tempText;
    els.popE2e.textContent = fmtMs(e2e.last);
    els.popVcm.textContent = fmtMs(vcm.last);
    els.popWake.textContent = fmtMs(wake.last);
    els.popCpu.textContent = cpuText;
    els.popRam.textContent = ramText;
    els.popTemp.textContent = tempText;
    els.popTurns.textContent = turns;
    els.popAccuracy.textContent = accuracyText;
    els.popFalseWakes.textContent = falseWakes;

    // diagnostics: counters + model facts
    els.diagTurns.textContent = turns;
    els.diagAccuracy.textContent = accuracyText;
    els.diagFalseWakes.textContent = falseWakes;
    els.diagRejection.textContent = turnsCount ? Math.round((ignoredCount / turnsCount) * 100) + "%" : "—";
    els.diagCpuLast.textContent = "CPU " + cpuText;
    els.diagRamLast.textContent = "RAM " + ramText;
    els.diagTempLast.textContent = "SoC " + tempText;
    if (msg.model) {
      document.getElementById("diag-model-name").textContent = msg.model.name || "—";
      document.getElementById("diag-model-params").textContent = msg.model.params ? (msg.model.params / 1e6).toFixed(2) + " M" : "—";
      document.getElementById("diag-model-version").textContent = msg.model.version || "—";
    }

    redrawDiagCharts();
  }

  /* ---------------- diagnostics charts (inline SVG) ---------------- */

  var SVG_NS = "http://www.w3.org/2000/svg";
  var DIAG_CHARTS = [
    { key: "e2e", svg: "diag-chart-e2e", hover: "diag-e2e-hover", unit: " ms", p50: "diag-e2e-p50", p90: "diag-e2e-p90" },
    { key: "vcm", svg: "diag-chart-vcm", hover: "diag-vcm-hover", unit: " ms", p50: "diag-vcm-p50", p90: "diag-vcm-p90" },
    { key: "wake", svg: "diag-chart-wake", hover: "diag-wake-hover", unit: " ms", p50: "diag-wake-p50", p90: "diag-wake-p90" },
    { key: "cpu", svg: "diag-chart-cpu", hover: "diag-cpu-hover", unit: "%" },
    { key: "ram", svg: "diag-chart-ram", hover: "diag-ram-hover", unit: " MB" },
    { key: "temp", svg: "diag-chart-temp", hover: "diag-temp-hover", unit: " °C" }
  ];

  function drawDiagChart(svgId, arr) {
    var svg = document.getElementById(svgId);
    if (!svg) return;
    var w = 320, h = 90, padL = 4, padR = 4, padT = 14, padB = 18;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    for (var i = 0; i <= 2; i++) {
      var y = padT + (h - padT - padB) * (i / 2);
      var gl = document.createElementNS(SVG_NS, "line");
      gl.setAttribute("class", "diag-grid-line");
      gl.setAttribute("x1", padL); gl.setAttribute("x2", w - padR);
      gl.setAttribute("y1", y.toFixed(1)); gl.setAttribute("y2", y.toFixed(1));
      svg.appendChild(gl);
    }
    if (!arr.length) return;
    var max = Math.max.apply(null, arr), min = Math.min.apply(null, arr);
    var range = Math.max(max - min, 1);
    var innerW = w - padL - padR, innerH = h - padT - padB;
    var step = arr.length > 1 ? innerW / (arr.length - 1) : 0;
    var pts = arr.map(function (v, i) {
      var x = padL + i * step;
      var y = padT + innerH - ((v - min) / range) * innerH;
      return x.toFixed(1) + "," + y.toFixed(1);
    }).join(" ");
    var poly = document.createElementNS(SVG_NS, "polyline");
    poly.setAttribute("class", "diag-line");
    poly.setAttribute("points", pts);
    svg.appendChild(poly);
    var maxLabel = document.createElementNS(SVG_NS, "text");
    maxLabel.setAttribute("class", "diag-axis-label");
    maxLabel.setAttribute("x", padL); maxLabel.setAttribute("y", padT - 3);
    maxLabel.textContent = Math.round(max);
    svg.appendChild(maxLabel);
    var minLabel = document.createElementNS(SVG_NS, "text");
    minLabel.setAttribute("class", "diag-axis-label");
    minLabel.setAttribute("x", padL); minLabel.setAttribute("y", h - 4);
    minLabel.textContent = Math.round(min);
    svg.appendChild(minLabel);
  }

  function redrawDiagCharts() {
    DIAG_CHARTS.forEach(function (c) {
      drawDiagChart(c.svg, hist[c.key]);
      if (c.p50) document.getElementById(c.p50).textContent = "p50 " + fmtMs(percentile(hist[c.key], 0.5));
      if (c.p90) document.getElementById(c.p90).textContent = "p90 " + fmtMs(percentile(hist[c.key], 0.9));
    });
  }

  DIAG_CHARTS.forEach(function (c) {
    var svg = document.getElementById(c.svg);
    var hoverEl = document.getElementById(c.hover);
    if (!svg || !hoverEl) return;
    function showAt(i) {
      var arr = hist[c.key];
      if (!arr.length) { hoverEl.textContent = " "; return; }
      i = Math.max(0, Math.min(arr.length - 1, i));
      hoverEl.textContent = Math.round(arr[i]) + c.unit;
    }
    svg.addEventListener("mousemove", function (ev) {
      var rect = svg.getBoundingClientRect();
      var ratio = rect.width ? (ev.clientX - rect.left) / rect.width : 0;
      showAt(Math.round(ratio * (hist[c.key].length - 1)));
    });
    svg.addEventListener("mouseleave", function () { hoverEl.textContent = " "; });
    svg.addEventListener("focus", function () { showAt(hist[c.key].length - 1); });
  });

  /* ---------------- system log ---------------- */

  function appendLog(level, text) {
    var li = document.createElement("li");
    li.className = level === "warn" ? "log--warn" : level === "error" ? "log--error" : "";
    var time = document.createElement("span");
    time.className = "log-time";
    time.textContent = new Date().toLocaleTimeString([], { hour12: false });
    li.appendChild(time);
    li.appendChild(document.createTextNode(text));
    els.logList.appendChild(li);
    while (els.logList.children.length > 30) els.logList.removeChild(els.logList.firstChild);
  }

  /* ---------------- turns: understood-sentence rebuild ---------------- */

  var INTENT_LIST = COMMAND_LIST;

  function pad2(n) { n = Number(n); return isNaN(n) ? "??" : (n < 10 ? "0" + n : String(n)); }

  function reasonText(turn) {
    var heard = turn.heard || {};
    var pct = typeof heard.prob === "number" ? " " + Math.round(heard.prob * 100) + "%" : "";
    if (heard.reason === "no_speech") return "Wake word without speech";
    if (heard.command === "OUT_OF_SCOPE") return "Not a command" + pct;
    if (heard.reason) return String(heard.reason).replace(/_/g, " ").replace(/^./, function (c) { return c.toUpperCase(); });
    return "Ignored" + pct;
  }

  function fmtLogTime(turn) {
    var d = turn && typeof turn.t === "number" ? new Date(turn.t * 1000) : new Date();
    return d.toLocaleTimeString([], { hour12: false });
  }

  function playClip(url) {
    if (!url) return;
    try {
      var audio = new Audio(url);
      audio.play().catch(function () { appendLog("warn", "Could not play " + url); });
    } catch (e) { appendLog("warn", "Could not play " + url); }
  }

  /* ---------------- turns: Activity table + Now card + Now activity list ---------------- */

  var turnRefs = {}; // turn id -> { tr, pickerTr, yesBtn, noBtn, picker }
  var currentTurnId = null;
  var NOW_ACTIVITY_MAX = 12;

  function mkEl(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function finishMarkAll(id) {
    var ref = turnRefs[id];
    if (ref) {
      ref.yesBtn.classList.add("is-marked");
      ref.noBtn.classList.add("is-marked");
      ref.picker.hidden = true;
      if (ref.pickerTr) ref.pickerTr.hidden = true;
    }
    if (currentTurnId === id) {
      els.exchangeMarkYes.classList.add("is-marked");
      els.exchangeMarkNo.classList.add("is-marked");
      els.exchangePicker.hidden = true;
    }
  }

  function buildIntentPicker(container, onPick) {
    container.innerHTML = "";
    INTENT_LIST.forEach(function (intentName) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = intentName;
      b.addEventListener("click", function () { onPick(intentName); });
      container.appendChild(b);
    });
  }

  function updateActivityEmptyState() {
    els.activityEmpty.hidden = els.activityTbody.children.length > 0;
  }

  function buildActivityRow(id, timeStr, isIgnored, heard, understood, confPct, reply) {
    var tr = document.createElement("tr");
    tr.dataset.turnId = id;
    if (isIgnored) tr.classList.add("is-ignored");

    tr.appendChild(mkEl("td", "cell-time", timeStr));

    var saidTd = mkEl("td", "cell-said");
    saidTd.textContent = isIgnored ? reasonText({ heard: heard }) : "“" + understood + "”";
    tr.appendChild(saidTd);

    tr.appendChild(mkEl("td", "cell-intent", isIgnored ? "—" : (heard.command || "—")));
    tr.appendChild(mkEl("td", "cell-conf", isIgnored ? "—" : confPct));

    var watsonTd = mkEl("td", "cell-said");
    watsonTd.textContent = reply ? "“" + (reply.text || "") + "”" : "—";
    tr.appendChild(watsonTd);

    var playTd = mkEl("td", "cell-actions");
    var yesBtn, noBtn, picker;
    if (heard.audio_url) {
      var playBtn = mkEl("button", "round-btn", "▶");
      playBtn.type = "button";
      playBtn.setAttribute("aria-label", "Play what the mic heard");
      playBtn.addEventListener("click", function () { playClip(heard.audio_url); });
      playTd.appendChild(playBtn);
    }
    if (reply && reply.audio_url) {
      var replyPlayBtn = mkEl("button", "round-btn", "▶");
      replyPlayBtn.type = "button";
      replyPlayBtn.setAttribute("aria-label", "Play reply");
      replyPlayBtn.addEventListener("click", function () { playClip(reply.audio_url); });
      playTd.appendChild(replyPlayBtn);
    }
    tr.appendChild(playTd);

    var markTd = mkEl("td", "cell-actions");
    if (!isIgnored) {
      yesBtn = mkEl("button", "round-btn round-btn--yes", "✓");
      yesBtn.type = "button";
      yesBtn.setAttribute("aria-label", "Mark this turn correct");
      noBtn = mkEl("button", "round-btn round-btn--no", "✗");
      noBtn.type = "button";
      noBtn.setAttribute("aria-label", "Mark this turn incorrect");
      markTd.appendChild(yesBtn);
      markTd.appendChild(noBtn);
    }
    tr.appendChild(markTd);

    var pickerTr = null;
    if (!isIgnored) {
      pickerTr = document.createElement("tr");
      pickerTr.hidden = true;
      var pickerTd = document.createElement("td");
      pickerTd.colSpan = 7;
      picker = mkEl("div", "intent-picker");
      buildIntentPicker(picker, function (intentName) {
        send({ type: "mark", turn_id: id, correct: false, intended_command: intentName });
        finishMarkAll(id);
      });
      pickerTd.appendChild(picker);
      pickerTr.appendChild(pickerTd);

      yesBtn.addEventListener("click", function () {
        send({ type: "mark", turn_id: id, correct: true });
        finishMarkAll(id);
      });
      noBtn.addEventListener("click", function () {
        picker.hidden = !picker.hidden;
        pickerTr.hidden = picker.hidden;
      });
    }

    return { tr: tr, pickerTr: pickerTr, yesBtn: yesBtn, noBtn: noBtn, picker: picker };
  }

  function appendNowActivityRow(id, timeStr, isIgnored, understood, reasonStr, heard, reply) {
    var li = document.createElement("li");
    li.className = "now-activity-row" + (isIgnored ? " now-activity-row--ignored" : "");
    if (isIgnored) {
      var igLine = mkEl("p", "now-activity-line");
      igLine.appendChild(mkEl("span", "now-activity-label", "Ignored"));
      igLine.appendChild(mkEl("span", "now-activity-text", reasonStr));
      li.appendChild(igLine);
    } else {
      var youLine = mkEl("p", "now-activity-line");
      youLine.appendChild(mkEl("span", "now-activity-label", "Understood as"));
      youLine.appendChild(mkEl("span", "now-activity-text", understood));
      li.appendChild(youLine);
      if (reply) {
        var wLine = mkEl("p", "now-activity-line");
        wLine.appendChild(mkEl("span", "now-activity-label", "Watson"));
        wLine.appendChild(mkEl("span", "now-activity-text", reply.text || ""));
        li.appendChild(wLine);
      }
    }
    li.appendChild(mkEl("span", "now-activity-time", timeStr));
    els.nowActivityList.appendChild(li);
    while (els.nowActivityList.children.length > NOW_ACTIVITY_MAX) {
      els.nowActivityList.removeChild(els.nowActivityList.firstChild);
    }
  }

  function updateExchangeCard(id, isIgnored, heard, understood, slotsText, confPct, reply) {
    els.exchangeEmpty.hidden = true;
    els.exchangeBody.hidden = false;

    if (isIgnored) {
      els.exchangeHeardText.textContent = reasonText({ heard: heard });
      els.exchangeHeardMeta.textContent = "Ignored";
      els.exchangeReplyText.textContent = reply ? reply.text || "" : "";
      els.exchangePlayHeard.hidden = !heard.audio_url;
      if (heard.audio_url) els.exchangePlayHeard.onclick = function () { playClip(heard.audio_url); };
      els.exchangePlayReply.hidden = true;
      els.exchangeMarkYes.hidden = true;
      els.exchangeMarkNo.hidden = true;
      els.exchangePicker.hidden = true;
      return;
    }

    els.exchangeHeardText.textContent = "“" + understood + "”";
    var metaBits = [];
    if (heard.command) metaBits.push(heard.command);
    if (confPct) metaBits.push(confPct);
    if (slotsText) metaBits.push(slotsText);
    els.exchangeHeardMeta.textContent = metaBits.join(" · ");

    els.exchangeReplyText.textContent = reply ? reply.text || "" : "";

    els.exchangePlayHeard.hidden = !heard.audio_url;
    if (heard.audio_url) els.exchangePlayHeard.onclick = function () { playClip(heard.audio_url); };
    els.exchangePlayReply.hidden = !(reply && reply.audio_url);
    if (reply && reply.audio_url) { var rUrl = reply.audio_url; els.exchangePlayReply.onclick = function () { playClip(rUrl); }; }

    els.exchangeMarkYes.hidden = false;
    els.exchangeMarkNo.hidden = false;
    els.exchangeMarkYes.classList.remove("is-marked");
    els.exchangeMarkNo.classList.remove("is-marked");
    els.exchangeMarkYes.onclick = function () {
      send({ type: "mark", turn_id: id, correct: true });
      finishMarkAll(id);
    };
    els.exchangeMarkNo.onclick = function () {
      els.exchangePicker.hidden = !els.exchangePicker.hidden;
    };
    buildIntentPicker(els.exchangePicker, function (intentName) {
      send({ type: "mark", turn_id: id, correct: false, intended_command: intentName });
      finishMarkAll(id);
    });
    els.exchangePicker.hidden = true;
  }

  function appendTurn(turn) {
    var id = turn.id != null ? String(turn.id) : "t" + Date.now() + Math.random();
    if (turnRefs[id]) return; // avoid dupes if a turn is resent

    var heard = turn.heard || {};
    var reply = turn.reply || null;
    var isIgnored = !reply || heard.accepted === false;

    turnsCount++;
    if (isIgnored) ignoredCount++;

    var timeStr = fmtLogTime(turn);
    var understood = isIgnored ? null : (heard.text || heard.command || "…");
    var slotsText = isIgnored ? "" : (heard.slot || "");
    var confPct = typeof heard.prob === "number" ? Math.round(heard.prob * 100) + "%" : "—";
    var reasonStr = isIgnored ? reasonText(turn) : "";

    var ref = buildActivityRow(id, timeStr, isIgnored, heard, understood, confPct, reply);
    els.activityTbody.appendChild(ref.tr);
    if (ref.pickerTr) els.activityTbody.appendChild(ref.pickerTr);
    turnRefs[id] = ref;
    updateActivityEmptyState();

    appendNowActivityRow(id, timeStr, isIgnored, understood, reasonStr, heard, reply);

    currentTurnId = id;
    updateExchangeCard(id, isIgnored, heard, understood, slotsText, confPct, reply);
  }

  els.clearActivity.addEventListener("click", function () {
    els.activityTbody.innerHTML = "";
    els.nowActivityList.innerHTML = "";
    turnRefs = {};
    currentTurnId = null;
    turnsCount = 0;
    ignoredCount = 0;
    updateActivityEmptyState();
    els.exchangeEmpty.hidden = false;
    els.exchangeBody.hidden = true;
  });

  updateActivityEmptyState();

})();
