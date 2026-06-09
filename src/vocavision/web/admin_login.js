const form = document.getElementById("admin-login-form");
const passwordInput = document.getElementById("admin-password");
const errorText = document.getElementById("admin-login-error");

async function login(event) {
  event.preventDefault();
  errorText.textContent = "";
  try {
    const response = await fetch("/api/admin/session", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        password: passwordInput.value,
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "登录失败，请检查密码。");
    }
    window.location.href = "/";
  } catch (error) {
    errorText.textContent = error.message;
  }
}

form.addEventListener("submit", login);
