
To create .vtis:
./vtkColorCodedDepthVolume C:\Users\vic-b\Documents\Victors\Projects\PhD\US_samples\US_samples\1\20 semanas\NRRD


To use paraview on the server:

Start the server on the remote machine:
Log in to your remote server via SSH and launch the server application:

Bash
pvserver

Set up an SSH Tunnel (Security Best Practice):
To ensure a secure connection (and to bypass potential firewall issues), create an SSH tunnel from your local machine to the server:

Bash
ssh -L 11111:localhost:11111 vbayer@ohpc


Connect your local ParaView:Open the ParaView application on your local machine.Go to File > Connect (or click the 'Connect' icon in the toolbar).  Click Add Server.Enter a name for the connection (e.g., "Remote Server") and ensure the Host is set to localhost and the Port is 11111.Click Configure and then Save.Select your new server configuration and click Connect.