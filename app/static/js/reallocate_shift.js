document.addEventListener('DOMContentLoaded', () => {
  const internBase = [10, 10];
  const residentBase = [10, 10];
  const specialistBase = [14, 10];
  const juniorBase = [2, 5, 2];
  const seniorBase = [1, 2, 1];

  function generateOptions(base) {
    const delta = Math.max(1, Math.floor(base * 0.1));
    const options = [];
    for (let i = base - delta; i <= base + delta; i++) {
      options.push(`<option value="${i}">${i}</option>`);
    }
    return options.join('');
  }

  function createShiftFields(role, base, keepTotal, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    const roleLower = role.toLowerCase();

    for (let i = 0; i < base.length; i++) {
      const shiftIndex = i + 1;  
      const label = `${role} Shift ${shiftIndex}:`;
      const name = `${roleLower}_shift_${shiftIndex}`;

      if (keepTotal && i === base.length - 1) {
        const total = base.reduce((a, b) => a + b, 0);
        let sum = 0;
        for (let j = 0; j < base.length - 1; j++) {
          const prevId = `${roleLower}_shift_${j + 1}`; 
          const field = document.getElementById(prevId);
          const val = field ? parseInt(field.value || base[j]) : base[j];
          sum += val;
        }
        const finalVal = Math.max(0, total - sum);

        container.innerHTML += `
          <label>${label}</label>
          <input type="number" name="${name}" id="${name}" readonly value="${finalVal}">
          <input type="hidden" name="${name}" id="${name}" value="${finalVal}">
        `;
      } else {
        container.innerHTML += `
          <label>${label}</label>
          <select name="${name}" id="${name}">
            ${generateOptions(base[i])}
          </select>
        `;
      }
    }
  }

  function recalculate(role, base) {
    const keepTotal = document.getElementById(`keep_total_${role.toLowerCase()}`).value === 'yes';
    if (!keepTotal) return;

    const roleLower = role.toLowerCase();
    const total = base.reduce((a, b) => a + b, 0);
    let sum = 0;
    for (let i = 0; i < base.length - 1; i++) {
      const id = `${roleLower}_shift_${i + 1}`; // ✅ 1부터 시작
      const val = parseInt(document.getElementById(id).value || 0);
      sum += val;
    }
    const finalInputId = `${roleLower}_shift_${base.length}`; // 마지막 index = length
    const finalInput = document.getElementById(finalInputId);
    if (finalInput) finalInput.value = Math.max(0, total - sum);
  }

  function setup(role, base, containerId) {
    const keepTotalEl = document.getElementById(`keep_total_${role.toLowerCase()}`);
    keepTotalEl.addEventListener('change', () => {
      createShiftFields(role, base, keepTotalEl.value === 'yes', containerId);
      setTimeout(() => recalculate(role, base), 50);
    });

    document.addEventListener('change', (e) => {
      if (e.target.name && e.target.name.startsWith(`${role.toLowerCase()}_shift_`)) {
        recalculate(role, base);
      }
    });

    createShiftFields(role, base, keepTotalEl.value === 'yes', containerId);
    setTimeout(() => recalculate(role, base), 50);
  }

  setup('intern', internBase, 'intern_shift_inputs');
  setup('resident', residentBase, 'resident_shift_inputs');
  setup('specialist', specialistBase, 'specialist_shift_inputs');
  setup('junior', juniorBase, 'junior_shift_inputs');
  setup('senior', seniorBase, 'senior_shift_inputs');
});
