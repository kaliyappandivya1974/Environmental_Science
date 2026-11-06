(() => {
  let notificationPrimed = false;
  let seenNotificationIds = new Set();
  const presetNames={normal:'Normal conditions',rain:'Heavy rainfall',river:'Rising river',air:'High air pollution',water:'Water quality anomaly'};

  function notificationMarkup(item) {
    const isUnread = !item.read_at;
    return `<div class="notification-item ${isUnread ? 'unread' : 'read'}"><i></i><div><b>${esc(item.title)}</b><small>${esc(item.message)} · ${esc(fmtDate(item.created_at))}</small><a href="#predictions" data-open-event="${esc(item.event_id)}">Open Risk Predictor ↗</a></div>${isUnread ? `<button type="button" data-read-notification="${item.notification_id}">Mark read</button>` : '<span class="small-tag">READ</span>'}</div>`;
  }

  function eventMarkup(item) {
    const p = item.prediction || {};
    const anomaly = p.anomaly;
    const flood = p.flood, fire = p.fire;
    const score = (label, value) => value ? `<span class="saved-score">${esc(label)} · <strong>${esc(value.category)}</strong> · ${fmtNum(value.risk_score * 100, 1)} score</span>` : '';
    const water = anomaly ? `<span class="saved-score">Water quality · <strong>${anomaly.is_anomaly ? 'Anomaly' : 'Normal'}</strong></span>` : '';
    return `<article class="saved-event" data-event-id="${esc(item.event_id)}"><div class="saved-event-head"><b>${esc(item.zone_type)} · Environmental prediction</b><span class="small-tag">${esc(item.source)}</span></div><span class="saved-event-meta">${esc(fmtDate(item.prediction_timestamp))} · ${esc(item.event_id)}</span><div class="saved-event-scores">${score('Flood', flood)}${score('Fire', fire)}${p.pm25_forecast_ug_m3 == null ? '' : `<span class="saved-score">PM2.5 · <strong>${fmtNum(p.pm25_forecast_ug_m3, 1)} µg/m³</strong></span>`}${water}</div><span class="saved-event-link">${esc(item.model_version || 'Model')} · ${esc(item.data_quality_status)}</span></article>`;
  }

  async function pollSharedEvents() {
    try {
      const [events, notificationData] = await Promise.all([api('/prediction-events?limit=30'), api('/notifications?limit=30')]);
      $('#saved-risk-events').innerHTML = events.length ? events.map(eventMarkup).join('') : '<div class="detail-empty">No saved prediction events yet. Run a reading or a Synthetic Lab scenario.</div>';
      $('#sync-status').textContent = `${events.length} saved events · Last synced ${new Date().toLocaleTimeString()}`;
      $('#top-alert-count').textContent = notificationData.unread_count;
      $('#alert-count').textContent = notificationData.unread_count;
      $('#notification-list').innerHTML = notificationData.notifications.length ? notificationData.notifications.map(notificationMarkup).join('') : '<div class="detail-empty">No in-app notifications yet.</div>';
      const currentIds = new Set(notificationData.notifications.map(n => Number(n.notification_id)));
      if (notificationPrimed) {
        for (const item of notificationData.notifications) {
          if (!seenNotificationIds.has(Number(item.notification_id)) && !item.read_at) toast(`${item.title} · ${item.zone_type} · ${item.source}`);
        }
      }
      notificationPrimed = true;
      seenNotificationIds = currentIds;
    } catch (error) {
      $('#sync-status').textContent = `Could not sync: ${error.message}`;
    }
  }
  window.pollSharedEvents = pollSharedEvents;

  function renderSettingsPage() {
    detailHeader('settings', 'Notification settings', 'Choose which saved synthetic risk events appear in persistent in-app notifications.', `<section class="card detail-card wide"><div class="section-kicker"><span class="kicker-mark teal">♧</span>IN-APP NOTIFICATIONS</div><h2>Notification rules</h2><form id="notification-settings-form"><label class="toggle-row"><span><b>In-app notifications</b><small>Keep qualifying risk events in the dashboard notification bell.</small></span><input id="notify-inapp" type="checkbox" checked></label><div class="settings-choice"><b>Hazard types</b><div><label><input type="checkbox" name="notify-hazard" value="flood" checked> Flood</label><label><input type="checkbox" name="notify-hazard" value="fire" checked> Forest fire</label><label><input type="checkbox" name="notify-hazard" value="water_quality" checked> Water quality anomaly</label></div></div><div class="settings-choice"><b>Risk severities</b><div><label><input type="checkbox" name="notify-severity" value="Low"> Low</label><label><input type="checkbox" name="notify-severity" value="Moderate"> Moderate</label><label><input type="checkbox" name="notify-severity" value="High" checked> High</label></div></div><div class="form-actions"><button type="submit" class="primary-button" id="save-notification-settings">Save preferences</button></div><p id="notification-settings-error" class="form-error" role="alert"></p></form></section>`);
    api('/notification-settings').then(data => {
      $('#notify-inapp').checked = data.in_app_enabled;
      document.querySelectorAll('[name="notify-hazard"]').forEach(el => el.checked = data.hazard_types.includes(el.value));
      document.querySelectorAll('[name="notify-severity"]').forEach(el => el.checked = data.severities.includes(el.value));
    }).catch(error => { $('#notification-settings-error').textContent = error.message; });
    $('#notification-settings-form').addEventListener('submit', async event => {
      event.preventDefault();
      const payload = {in_app_enabled:$('#notify-inapp').checked,
        hazard_types:[...document.querySelectorAll('[name="notify-hazard"]:checked')].map(e=>e.value),
        severities:[...document.querySelectorAll('[name="notify-severity"]:checked')].map(e=>e.value)};
      const button=$('#save-notification-settings');button.disabled=true;$('#notification-settings-error').textContent='';
      try { await api('/notification-settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});toast('In-app notification preferences saved'); }
      catch(error) { $('#notification-settings-error').textContent=error.message;toast(error.message,true); }
      finally {button.disabled=false;}
    });
  }

  const originalRenderDetail=window.renderDetail;
  window.renderDetail=function(page){if(page==='settings'){renderSettingsPage();return;}return originalRenderDetail(page);};
  if(location.hash==='#settings')renderSettingsPage();

  $('#notification-button').addEventListener('click', async () => {
    const panel=$('#notification-popover');const open=panel.classList.contains('hidden');panel.classList.toggle('hidden',!open);$('#notification-button').setAttribute('aria-expanded',String(open));if(open)await pollSharedEvents();
  });
  $('#notification-list').addEventListener('click',async event=>{
    const read=event.target.closest('[data-read-notification]');
    if(read){read.disabled=true;try{await api(`/notifications/${read.dataset.readNotification}/read`,{method:'POST'});await pollSharedEvents();}catch(error){toast(error.message,true);}return;}
    const open=event.target.closest('[data-open-event]');
    if(open){const eventId=open.dataset.openEvent,item=event.target.closest('.notification-item'),button=item.querySelector('[data-read-notification]');if(button)await api(`/notifications/${button.dataset.readNotification}/read`,{method:'POST'});$('#notification-popover').classList.add('hidden');$('#notification-button').setAttribute('aria-expanded','false');showPage('predictions');await pollSharedEvents();document.querySelector(`[data-event-id="${CSS.escape(eventId)}"]`)?.scrollIntoView({behavior:'smooth',block:'center'});}
  });
  $('#mark-all-read').addEventListener('click',async()=>{try{await api('/notifications/read-all',{method:'POST'});await pollSharedEvents();}catch(error){toast(error.message,true);}});
  $('#detail-content').addEventListener('click',event=>{const link=event.target.closest('[data-event-details]');if(link){event.preventDefault();showPage('predictions');}});
  document.querySelectorAll('[data-simulation]').forEach(button=>button.addEventListener('click',async()=>{
    const preset=PRESETS[button.dataset.simulation];if(!preset)return;
    document.querySelectorAll('[data-simulation]').forEach(item=>item.disabled=true);button.querySelector('.scenario-run').textContent='Running existing models…';
    const reading={...preset,timestamp:new Date().toISOString()};
    try{const result=await api('/predict',{method:'POST',headers:{'Content-Type':'application/json','Idempotency-Key':crypto.randomUUID()},body:JSON.stringify(reading)});
      $('#simulation-result').innerHTML=`<div class="section-kicker"><span class="kicker-mark emerald">✓</span>SAVED PREDICTION EVENT</div><h2>${esc(reading.zone_type)} · ${esc(fmtDate(result.prediction_timestamp))}</h2><div class="saved-event-scores"><span class="saved-score">Flood · <strong>${esc(result.flood.category)}</strong> · ${fmtNum(result.flood.risk_score*100,1)} score</span><span class="saved-score">Fire · <strong>${esc(result.fire.category)}</strong> · ${fmtNum(result.fire.risk_score*100,1)} score</span><span class="saved-score">PM2.5 · <strong>${fmtNum(result.pm25_forecast_ug_m3,1)} µg/m³</strong></span><span class="saved-score">Water · <strong>${esc(result.anomaly.category)}</strong></span></div><p class="muted">Event ${esc(result.event_id)} · ${esc(result.notification_processing_status)} · synthetic demonstration</p><a class="inline-link" href="#predictions" data-go-risk>Open Risk Predictor ↗</a>`;
      toast(`Scenario saved · ${result.event_id}`);await pollSharedEvents();
    }catch(error){$('#simulation-result').innerHTML=`<div class="detail-empty error">Simulation failed: ${esc(error.message)}</div>`;toast(error.message,true);}
    finally{document.querySelectorAll('[data-simulation]').forEach(item=>item.disabled=false);button.innerHTML=`<span class="scenario-icon">↻</span><small>SCENARIO</small><b>${esc(presetNames[button.dataset.simulation])}</b><p>Run the scenario again using the model API.</p><span class="scenario-run">Run scenario <i>↗</i></span>`;}
  }));
  $('#simulation-result').addEventListener('click',event=>{if(event.target.closest('[data-go-risk]')){event.preventDefault();showPage('predictions');}});
  pollSharedEvents();setInterval(pollSharedEvents,8000);
})();
